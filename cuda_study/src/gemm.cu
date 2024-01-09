
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <time.h>
#include "util_func.h"

#define THREAD_PER_BLOCK 1024
#define WARP_SIZE 32
#define M 2048
#define N 2048
#define K 2048

#define BM 64
#define BN 64
#define BK 8
#define RM 4
#define RN 4

#define OFFSET(row, col, matrix_col_num) ((row) * (matrix_col_num) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/******************
总体优化思路：
1、提高计算访存比
2、近存计算，减少访存延时
3、
*******************/

__global__ void kernel_gemm0(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,int m,int n,int k){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx < n && idy < m){
		float result = 0;
		for(int i = 0; i < k; i++){
			result += d_matrix_a[idy * k + i] * d_matrix_b[i * n + idx];
		}
		d_matrix_c[idy * n + idx] = result;
	}
}

/*
Naive的计算方式，每个线程负责计算C矩阵的其中一个数,总共需要M*N个线程
每个线程需要加载A矩阵的一行以及B矩阵的一列，共K+K个数，因此总共需要对全局内存进行M*N*2K个读操作，以及M*N次写操作
*/
void gemm_compute0(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,float *h_matrix_c,float *h_matrix_c_gpu,int bcheck){
	int blockdim_x = 32;
	int blockdim_y = 32;
	/*
	线程布局式采用二维网格的方式，每个BLOCK内的线程为(blockdim_y,blockdim_x)
	grid的布局为((M + blockdim_y - 1) / blockdim_y,(N + blockdim_x - 1) / blockdim_x)
	*/
	dim3 block(blockdim_x,blockdim_y);
	dim3 grid((N + blockdim_x - 1) / blockdim_x,(M + blockdim_y - 1) / blockdim_y);
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	kernel_gemm0<<<grid,block>>>(d_matrix_a,d_matrix_b,d_matrix_c,M,N,K);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime = 0.0;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("gemm_naive cudaProcElapsed = %f\n",elapsedTime / 1000);
	
	cudaMemcpy(h_matrix_c_gpu,d_matrix_c,sizeof(float) * M * N,cudaMemcpyDeviceToHost);
	if(bcheck){
		check(h_matrix_c,h_matrix_c_gpu,M,N);
	}
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

/*
从共享内存中读取子矩阵进行矩阵的乘积
*/
__device__ void sub_matrix_multiply_0(float result[RM][RN],float sub_a[BM][BK],float sub_b[BK][BN]){
	for(int compute_i = 0; compute_i < RM; compute_i++){
		for(int compute_j = 0; compute_j < RN; compute_j++){
			for(int compute_k = 0; compute_k < BK; compute_k++){
				result[compute_i][compute_j] += sub_a[threadIdx.y * RM + compute_i][compute_k] * sub_b[compute_k][threadIdx.x * RN + compute_j];
			}
		}
	}
}

/*
优化：同样的优化思路，将A,B的子矩阵再次划分为子矩阵，然后迭代将第二级子矩阵依次从共享内存中加载到寄存器中，
从寄存器中进行第二级子矩阵的乘积，从而减少对共享内存的访问次数
*/
__device__ void sub_matrix_multiply_1(float result[RM][RN],float sub_a[BM][BK],float sub_b[BK][BN]){
	//将sub_a,sub_b划分为BK个子矩阵，sub_a每次读取一列，sub_b每次读取一行，总共迭代BK次
	float reg_sub_a[RM];
	float reg_sub_b[RN];
	for(int k = 0; k < BK; k++){
		for(int i = 0; i < RM; i++){
			reg_sub_a[i] = sub_a[threadIdx.y * RM + i][k];
		}
		for(int i = 0; i < RN; i++){
			reg_sub_b[i] = sub_b[k][threadIdx.x * RN + i];
		}
		for(int i = 0; i < RM;i++){
			for(int j = 0; j < RN; j++){
				result[i][j] += reg_sub_a[i] * reg_sub_b[j];
			}
		}
	}
}

/*
说明：以kernel_gemm1_开始的核函数支持任意大小的矩阵相乘，因此会考虑在矩阵分块时不能分成整数块的情况
优化1：
Naive算法中存在的问题：
在计算C时，每个线程都要去全局内存中加载A矩阵的一行和B矩阵的一列，这里存在两个问题：
1、从全局内存中加载的时延很大。
2、因为矩阵是采用行存储，在加载B矩阵时从列方向加载，因此缓存命中率很低，甚至没有命中
3、每次加载的数据并没有充分使用，因为根据矩阵的乘法，加载A矩阵的一行可以B矩阵的所有列进行向量乘。

优化思路：
1、将数据加载到共享内存中，然后在共享内存中多次使用，以减少到全局内存的访问。
因为共享内存空间有限，因此需要对矩阵进行分块，每次只加载一小块，加载一小块的同时也提高了缓存的命中率。

具体方式：
将C切成多个小块，每个小块的大小为BM * BN,每个小块由一个BLOCK完成计算。总共需要的BLOCK数BLOCK_NUM = ((M + BM - 1) / BM) * ((N + BN - 1) / BN)
因此对于第(i,j)个BLOCK，需要读取A矩阵的子矩阵SUB_A = A[i*BM : (i + 1)*BM -1][0 ：N],即第i*BM行到(i + 1)*BM -1行数据
需要读取B矩阵的子矩阵SUB_B = B[0 : K][j*BN : (j + 1)*BN -1],即第j*BN 到(j + 1)*BN -1列数据
然后计算SUB_A * SUB_B

由于SUB_A和SUB_B依然很大(SUB_A的宽度和SUB_B的高度都为K，K可能是一个较大的值)，无法直接加载到全局变量中，因此需要继续对SUB_A和SUB_B进行切分。
切分的方法为：
SUB_A在横向切分，每个小块的大小为BM*BK,切分的块数SUB1_SPLIT_NUM = (K + BK -1)/BK
SUB_B在竖向切分，每个小块的大小为BK*BN,切分的块数同样为SUB1_SPLIT_NUM
因此计算C矩阵的BM*BN子块，需要进行SUB1_SPLIT_NUM次SUB_A的子块SUB_SUB_A[BM][BK]和SUB_B的子块SUB_SUB_B[BK][BN]相乘，把这
SUB1_SPLIT_NUM次的计算结果累加。
接下来统计对全局内存的访问：
每个BLOCK从全局内存加载的数据量为SUB_A+SUB_B的数据量，即K*BM + K * BN,因此总共需要读取的次数为：
BLOCK_NUM * (K*BM + K * BN) = ((M + BM - 1) / BM) * ((N + BN - 1) / BN) * (K * (BM + BN)) ~= M*N*K(1/BN + 1/BM)
为Naive方法的读操作数的0.5 *(1/BN + 1/BM)倍

最后考虑BLOCK内的线程安排，由于GPU的一个BLOCK最大只支持1024个线程，而BM*BN可能超过1024，因此需要对BM*BN继续分块，每个线程负责其中一块的计算。
假设分块的大小为RM*RN，则一个BLOCK内线程的布局为：((BM + RM - 1)/RM,(BN + RN - 1) / RN),注意线程总线不能超过1024
*/
__global__ void kernel_gemm1_0(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,int m,int n,int k){
	//分配共享内存，用于加载a和b的各一小块子矩阵
	__shared__ float sub_a[BM][BK];
	__shared__ float sub_b[BK][BN];
	float result[RM][RN] = {0};//每个线程计算rm*rn个数据
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	//TEST
	//if(blockIdx.x != 0 || blockIdx.y != 0 || idx != 325) return;
	//printf("idx = %d\n",idx);
								 
	/*将C矩阵m*n，切割成bm*bn的小块，每个block计算其中一小块*/
	//计算当前BLOCK要处理的a矩阵的首行号和尾行号
	int a_start_row = blockIdx.y * BM;//每个BLOCK处理bm行
	int a_end_row = a_start_row + BM - 1;
	if(a_end_row >= m) a_end_row = m - 1;
	
	//计算当前BLOCK要处理的b矩阵的首列号和尾列号
	int b_start_col = blockIdx.x * BN;//每个BLOCK处理bn列
	int b_end_col = b_start_col + BN - 1;
	if(b_end_col >= n) b_end_col = n - 1;
	
	/*无法将SUB_A = A[a_start_row:a_end_row][0 ：n]，SUB_B = B[0 : k][b_start_col:b_end_col]一次般入共享内存，因此对SUB_A进行横向上的切块，
	  对SUB_B进行竖向方向的切块。每次从SUB_A和SUB_B上取一个小块，然后计算这两个小块的矩阵乘法：A矩阵每块的大小bm*bk,B矩阵每块的大小为bk*bn*/
	//根据分块信息，计算总共需要进行的计算迭代数
	int block_compute_num = (k + BK - 1) / BK;
	
	//依次加载一小块a和一小块b到共享内存中，然后启动计算
	for(int i = 0; i < block_compute_num; i++){
		//计算要加载的a矩阵的首列号和尾列号
		int a_start_col = i * BK;//每次迭代处理bk列
		int a_end_col = a_start_col + BK - 1;
		if(a_end_col >= k) a_end_col = k - 1;
		
		//计算要加载的b矩阵的首行号和尾行号,等于a矩阵的首列号和尾列号
		int b_start_row = a_start_col;
		int b_end_row = a_end_col;
		
		//接下来考虑a[a_start_row:a_end_row][a_start_col:a_end_col],b[b_start_row:b_end_row][b_start_col:b_end_col]这两个小矩阵的加载方式
		//为了减少访存指令，每个线程load一个float4,也就是4个数据，计算每个线程需要Load的次数
		int block_load_num = blockDim.x * blockDim.y * 4;
		int load_a_gmem_num = (BM * BK) / block_load_num;
		//至少读取一次
		if(load_a_gmem_num == 0) load_a_gmem_num = 1;
		int load_b_gmem_num = (BK * BN) / block_load_num;
		//至少读取一次
		if(load_b_gmem_num == 0) load_b_gmem_num = 1;
		//printf("i = %d,a[%d:%d][%d:%d],b[%d:%d][%d:%d]\n",i,a_start_row,a_end_row,a_start_col,a_end_col,b_start_row,b_end_row,b_start_col,b_end_col);
		//首先将全局内存中的数据Load进寄存器
		//float ldg_a_reg[4] = {0};
		//float ldg_b_reg[4] = {0};
		//读取A的子矩阵
		__syncthreads();
		for(int j = 0; j < load_a_gmem_num; j++){
			//当前线程读取的数据相对于min_a子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_a[load_sub_a_row = offset / bk][load_sub_a_col = offset % bk] 
			//对应到原始a矩阵的坐标为a[a_start_row + min_a_row][a_start_col + min_a_col]
			int load_sub_a_row = (j * block_load_num + idx * 4) / BK;
			int load_sub_a_col = (j * block_load_num + idx * 4) % BK;
			int load_a_row = a_start_row + load_sub_a_row;
			int load_a_col = a_start_col + load_sub_a_col;
			//printf("a block_load_num = %d,j = %d,load_sub_a_row = %d,%d,%d\n",block_load_num,j,load_sub_a_row,load_a_row,load_a_col);
			if(load_a_row <= a_end_row && load_a_col <= a_end_col){
				//注意每次是加载4个字节，如果a的列数不是4的倍数，则有可能多读
				//如果列数不是4的倍数，使用float4加载会失败，不满足对齐要求，因此 针对普通场景，不使用float4加载
				//FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(d_matrix_a[OFFSET(load_a_row,load_a_col,k)]);
				//将加载到寄存器的值写入共享内存，这里暂时没有考虑bank冲突,(a_end_col - a_start_col)表示实际要写入数据，有可能不需要4字节
				for(int t = 0; t < 4; t++){
					if(t <= a_end_col - a_start_col){
						sub_a[load_sub_a_row][load_sub_a_col + t] = d_matrix_a[OFFSET(load_a_row,load_a_col,k) + t];
					}else{
						sub_a[load_sub_a_row][load_sub_a_col + t] = 0;
					}
				}
			}else if(load_sub_a_row < BM && load_sub_a_col < BK){
				//超出原矩阵的范围，则对共享内存位置填充为0
				sub_a[load_sub_a_row][load_sub_a_col] = 0;
				sub_a[load_sub_a_row][load_sub_a_col + 1] = 0;
				sub_a[load_sub_a_row][load_sub_a_col + 2] = 0;
				sub_a[load_sub_a_row][load_sub_a_col + 3] = 0;
			}
		}
		//读取B的子矩阵
		for(int j = 0; j < load_b_gmem_num; j++){
			//当前线程读取的数据相对于min_b子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_b[load_sub_b_row = offset / bn][load_sub_b_col = offset % bn] 
			//对应到原始b矩阵的坐标为b[b_start_row + min_b_row][b_start_col + min_b_col]
			int load_sub_b_row = (j * block_load_num + idx * 4) / BN;
			int load_sub_b_col = (j * block_load_num + idx * 4) % BN;
			int load_b_row = b_start_row + load_sub_b_row;
			int load_b_col = b_start_col + load_sub_b_col;
			//printf("b %d,%d\n",load_b_row,load_b_col);
			if(load_b_row <= b_end_row && load_b_col <= b_end_col){
				//注意每次是加载4个字节，如果b的列数不是4的倍数，则有可能多读
				//FETCH_FLOAT4(ldg_b_reg[0]) = FETCH_FLOAT4(d_matrix_b[OFFSET(load_b_row,load_b_col,n)]);
				//将加载到寄存器的值写入共享内存，这里暂时没有考虑bank冲突,(b_end_col - b_start_col)表示实际要写入数据，有可能不需要4字节
				for(int t = 0; t < 4; t++){
					if(t <= a_end_col - a_start_col){
						sub_b[load_sub_b_row][load_sub_b_col + t] = d_matrix_b[OFFSET(load_b_row,load_b_col,n) + t];
					}else{
						sub_b[load_sub_b_row][load_sub_b_col + t] = 0;
					}					
				}
			}else if(load_sub_b_row < BK && load_sub_b_col < BN){
				//超出原矩阵的范围，则对共享内存位置填充为0
				sub_b[load_sub_b_row][load_sub_b_col] = 0;
				sub_b[load_sub_b_row][load_sub_b_col + 1] = 0;
				sub_b[load_sub_b_row][load_sub_b_col + 2] = 0;
				sub_b[load_sub_b_row][load_sub_b_col + 3] = 0;
			}
		}
		__syncthreads();
		//加载完成后，启动计算，计算结果保存在寄存器中
		/*从sub_a和sub_b中再取一个子矩阵相乘,sub_a的子矩阵大小为rm * bk ,sub_b的子矩阵大小为bk * rn
		,当前线程处理sub_a横方向上的第threadIdx.y个子块，处理sub_b竖方向上的第threadIdx.x个子块*/
		sub_matrix_multiply_0(result,sub_a,sub_b);
	}
	//将每个线程的计算结果写回全局内存
	//每个线程处理的子块，对应到原始C矩阵的坐标为[a_start_row + threadIdx.y * rm][b_start_col + threadIdx.x * rn]
	int save_c_row = a_start_row + threadIdx.y * RM;
	int save_c_col = b_start_col + threadIdx.x * RN;
	for(int i = 0; i < RM ; i++){
		if((save_c_row + i) < m){ 
			for(int j = 0; j < RN ; j++){
				if((save_c_col + j) < n){
					d_matrix_c[OFFSET(save_c_row + i,save_c_col + j,n)] = result[i][j];
				}
			}
		}
	}
}

__global__ void kernel_gemm1_1(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,int m,int n,int k){
	//分配共享内存，用于加载a和b的各一小块子矩阵
	__shared__ float sub_a[BM][BK];
	__shared__ float sub_b[BK][BN];
	float result[RM][RN] = {0};//每个线程计算rm*rn个数据
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	//TEST
	//if(blockIdx.x != 0 || blockIdx.y != 0 || idx != 325) return;
	//printf("idx = %d\n",idx);
								 
	/*将C矩阵m*n，切割成bm*bn的小块，每个block计算其中一小块*/
	//计算当前BLOCK要处理的a矩阵的首行号和尾行号
	int a_start_row = blockIdx.y * BM;//每个BLOCK处理bm行
	int a_end_row = a_start_row + BM - 1;
	if(a_end_row >= m) a_end_row = m - 1;
	
	//计算当前BLOCK要处理的b矩阵的首列号和尾列号
	int b_start_col = blockIdx.x * BN;//每个BLOCK处理bn列
	int b_end_col = b_start_col + BN - 1;
	if(b_end_col >= n) b_end_col = n - 1;
	
	/*无法将SUB_A = A[a_start_row:a_end_row][0 ：n]，SUB_B = B[0 : k][b_start_col:b_end_col]一次般入共享内存，因此对SUB_A进行横向上的切块，
	  对SUB_B进行竖向方向的切块。每次从SUB_A和SUB_B上取一个小块，然后计算这两个小块的矩阵乘法：A矩阵每块的大小bm*bk,B矩阵每块的大小为bk*bn*/
	//根据分块信息，计算总共需要进行的计算迭代数
	int block_compute_num = (k + BK - 1) / BK;
	
	//依次加载一小块a和一小块b到共享内存中，然后启动计算
	for(int i = 0; i < block_compute_num; i++){
		//计算要加载的a矩阵的首列号和尾列号
		int a_start_col = i * BK;//每次迭代处理bk列
		int a_end_col = a_start_col + BK - 1;
		if(a_end_col >= k) a_end_col = k - 1;
		
		//计算要加载的b矩阵的首行号和尾行号,等于a矩阵的首列号和尾列号
		int b_start_row = a_start_col;
		int b_end_row = a_end_col;
		
		//接下来考虑a[a_start_row:a_end_row][a_start_col:a_end_col],b[b_start_row:b_end_row][b_start_col:b_end_col]这两个小矩阵的加载方式
		//为了减少访存指令，每个线程load一个float4,也就是4个数据，计算每个线程需要Load的次数
		int block_load_num = blockDim.x * blockDim.y * 4;
		int load_a_gmem_num = (BM * BK) / block_load_num;
		//至少读取一次
		if(load_a_gmem_num == 0) load_a_gmem_num = 1;
		int load_b_gmem_num = (BK * BN) / block_load_num;
		//至少读取一次
		if(load_b_gmem_num == 0) load_b_gmem_num = 1;
		//printf("i = %d,a[%d:%d][%d:%d],b[%d:%d][%d:%d]\n",i,a_start_row,a_end_row,a_start_col,a_end_col,b_start_row,b_end_row,b_start_col,b_end_col);
		//首先将全局内存中的数据Load进寄存器
		//float ldg_a_reg[4] = {0};
		//float ldg_b_reg[4] = {0};
		//读取A的子矩阵
		__syncthreads();
		for(int j = 0; j < load_a_gmem_num; j++){
			//当前线程读取的数据相对于min_a子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_a[load_sub_a_row = offset / bk][load_sub_a_col = offset % bk] 
			//对应到原始a矩阵的坐标为a[a_start_row + min_a_row][a_start_col + min_a_col]
			int load_sub_a_row = (j * block_load_num + idx * 4) / BK;
			int load_sub_a_col = (j * block_load_num + idx * 4) % BK;
			int load_a_row = a_start_row + load_sub_a_row;
			int load_a_col = a_start_col + load_sub_a_col;
			//printf("a block_load_num = %d,j = %d,load_sub_a_row = %d,%d,%d\n",block_load_num,j,load_sub_a_row,load_a_row,load_a_col);
			if(load_a_row <= a_end_row && load_a_col <= a_end_col){
				//注意每次是加载4个字节，如果a的列数不是4的倍数，则有可能多读
				//如果列数不是4的倍数，使用float4加载会失败，不满足对齐要求，因此 针对普通场景，不使用float4加载
				//FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(d_matrix_a[OFFSET(load_a_row,load_a_col,k)]);
				//将加载到寄存器的值写入共享内存，这里暂时没有考虑bank冲突,(a_end_col - a_start_col)表示实际要写入数据，有可能不需要4字节
				for(int t = 0; t < 4; t++){
					if(t <= a_end_col - a_start_col){
						sub_a[load_sub_a_row][load_sub_a_col + t] = d_matrix_a[OFFSET(load_a_row,load_a_col,k) + t];
					}else{
						sub_a[load_sub_a_row][load_sub_a_col + t] = 0;
					}
				}
			}else if(load_sub_a_row < BM && load_sub_a_col < BK){
				//超出原矩阵的范围，则对共享内存位置填充为0
				sub_a[load_sub_a_row][load_sub_a_col] = 0;
				sub_a[load_sub_a_row][load_sub_a_col + 1] = 0;
				sub_a[load_sub_a_row][load_sub_a_col + 2] = 0;
				sub_a[load_sub_a_row][load_sub_a_col + 3] = 0;
			}
		}
		//读取B的子矩阵
		for(int j = 0; j < load_b_gmem_num; j++){
			//当前线程读取的数据相对于min_b子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_b[load_sub_b_row = offset / bn][load_sub_b_col = offset % bn] 
			//对应到原始b矩阵的坐标为b[b_start_row + min_b_row][b_start_col + min_b_col]
			int load_sub_b_row = (j * block_load_num + idx * 4) / BN;
			int load_sub_b_col = (j * block_load_num + idx * 4) % BN;
			int load_b_row = b_start_row + load_sub_b_row;
			int load_b_col = b_start_col + load_sub_b_col;
			//printf("b %d,%d\n",load_b_row,load_b_col);
			if(load_b_row <= b_end_row && load_b_col <= b_end_col){
				//注意每次是加载4个字节，如果b的列数不是4的倍数，则有可能多读
				//FETCH_FLOAT4(ldg_b_reg[0]) = FETCH_FLOAT4(d_matrix_b[OFFSET(load_b_row,load_b_col,n)]);
				//将加载到寄存器的值写入共享内存，这里暂时没有考虑bank冲突,(b_end_col - b_start_col)表示实际要写入数据，有可能不需要4字节
				for(int t = 0; t < 4; t++){
					if(t <= a_end_col - a_start_col){
						sub_b[load_sub_b_row][load_sub_b_col + t] = d_matrix_b[OFFSET(load_b_row,load_b_col,n) + t];
					}else{
						sub_b[load_sub_b_row][load_sub_b_col + t] = 0;
					}					
				}
			}else if(load_sub_b_row < BK && load_sub_b_col < BN){
				//超出原矩阵的范围，则对共享内存位置填充为0
				sub_b[load_sub_b_row][load_sub_b_col] = 0;
				sub_b[load_sub_b_row][load_sub_b_col + 1] = 0;
				sub_b[load_sub_b_row][load_sub_b_col + 2] = 0;
				sub_b[load_sub_b_row][load_sub_b_col + 3] = 0;
			}
		}
		__syncthreads();
		//加载完成后，启动计算，计算结果保存在寄存器中
		/*从sub_a和sub_b中再取一个子矩阵相乘,sub_a的子矩阵大小为rm * bk ,sub_b的子矩阵大小为bk * rn
		,当前线程处理sub_a横方向上的第threadIdx.y个子块，处理sub_b竖方向上的第threadIdx.x个子块*/
		sub_matrix_multiply_1(result,sub_a,sub_b);
	}
	//将每个线程的计算结果写回全局内存
	//每个线程处理的子块，对应到原始C矩阵的坐标为[a_start_row + threadIdx.y * rm][b_start_col + threadIdx.x * rn]
	int save_c_row = a_start_row + threadIdx.y * RM;
	int save_c_col = b_start_col + threadIdx.x * RN;
	for(int i = 0; i < RM ; i++){
		if((save_c_row + i) < m){ 
			for(int j = 0; j < RN ; j++){
				if((save_c_col + j) < n){
					d_matrix_c[OFFSET(save_c_row + i,save_c_col + j,n)] = result[i][j];
				}
			}
		}
	}
}



/*和kernel_gemm1_0一致，只是对矩阵的大小有要求，即能分成整数块，因此此函数相比kernel_gemm1_0只是去除了所有关于
矩阵边界的判断处理，性能会稍优于kernel_gemm1_0*/
__global__ void kernel_gemm2_0(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,int m,int n,int k){
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	
	//分配共享内存，用于加载a和b的各一小块子矩阵
	__shared__ float sub_a[BM][BK];
	__shared__ float sub_b[BK][BN];
	float result[RM][RN] = {0};//每个线程计算rm*rn个数据
	
	//TEST
	//if(blockIdx.x != 0 || blockIdx.y != 0 || idx != 5) return;
	//printf("idx = %d\n",idx);
								 
	/*将C矩阵m*n，切割成bm*bn的小块，每个block计算其中一小块*/
	//计算当前BLOCK要处理的a矩阵的首行号和尾行号
	int a_start_row = blockIdx.y * BM;//每个BLOCK处理bm行

	//计算当前BLOCK要处理的b矩阵的首列号和尾列号
	int b_start_col = blockIdx.x * BN;//每个BLOCK处理bn列
	
	/*无法将SUB_A = A[a_start_row:a_end_row][0 ：n]，SUB_B = B[0 : k][b_start_col:b_end_col]一次般入共享内存，因此对SUB_A进行横向上的切块，
	  对SUB_B进行竖向方向的切块。每次从SUB_A和SUB_B上取一个小块，然后计算这两个小块的矩阵乘法：A矩阵每块的大小bm*bk,B矩阵每块的大小为bk*bn*/
	//根据分块信息，计算总共需要进行的计算迭代数
	int block_compute_num = (k + BK - 1) / BK;
	
	//接下来考虑a[a_start_row:a_end_row][a_start_col:a_end_col],b[b_start_row:b_end_row][b_start_col:b_end_col]这两个小矩阵的加载方式
	//为了减少访存指令，每个线程load一个float4,也就是4个数据，计算每个线程需要Load的次数
	int block_load_num = blockDim.x * blockDim.y * 4;
	int load_a_gmem_num = (BM * BK + block_load_num - 1) / block_load_num;
	int load_b_gmem_num = (BK * BN + block_load_num - 1) / block_load_num;
	//printf("load_a_gmem_num = %d,load_b_gmem_num = %d\n",load_a_gmem_num,load_b_gmem_num);
	//依次加载一小块a和一小块b到共享内存中，然后启动计算
	for(int i = 0; i < block_compute_num; i++){
		//计算要加载的a矩阵的首列号和尾列号
		int a_start_col = i * BK;//每次迭代处理bk列
		
		//计算要加载的b矩阵的首行号和尾行号,等于a矩阵的首列号和尾列号
		int b_start_row = a_start_col;
#if 1		
		__syncthreads();//遗留问题：这里必须要有同步语句，否则会出现A子矩阵读取不正确问题，没有搞清楚什么原因
		//读取A的子矩阵
		for(int j = 0; j < load_a_gmem_num; j++){
			//当前线程读取的数据相对于sub_a子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_a[load_sub_a_row = offset / bk][load_sub_a_col = offset % bk] 
			//对应到原始a矩阵的坐标为a[a_start_row + min_a_row][a_start_col + min_a_col]
			int load_sub_a_row = (j * block_load_num + idx * 4) / BK;
			int load_sub_a_col = (j * block_load_num + idx * 4) % BK;
			if(load_sub_a_row < BM && load_sub_a_col < BK){
				//printf("%d,%d,%d,%d\n",load_sub_a_row,load_sub_a_col,a_start_row + load_sub_a_row,a_start_col + load_sub_a_col);
				FETCH_FLOAT4(sub_a[load_sub_a_row][load_sub_a_col]) = \
					FETCH_FLOAT4(d_matrix_a[OFFSET(a_start_row + load_sub_a_row,a_start_col + load_sub_a_col,k)]);
			}
		}
		//读取B的子矩阵
		for(int j = 0; j < load_b_gmem_num; j++){
			//当前线程读取的数据相对于sub_b子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_b[load_sub_b_row = offset / bn][load_sub_b_col = offset % bn] 
			//对应到原始b矩阵的坐标为b[b_start_row + min_b_row][b_start_col + min_b_col]
			int load_sub_b_row = (j * block_load_num + idx * 4) / BN;
			int load_sub_b_col = (j * block_load_num + idx * 4) % BN;
			if(load_sub_b_row < BK && load_sub_b_col < BN){
				FETCH_FLOAT4(sub_b[load_sub_b_row][load_sub_b_col]) = \
					FETCH_FLOAT4(d_matrix_b[OFFSET(b_start_row + load_sub_b_row,b_start_col + load_sub_b_col,n)]);
			}
		}
#endif
		__syncthreads();
		//加载完成后，启动计算，计算结果保存在寄存器中
		/*每个线程从sub_a和sub_b中再取一个子矩阵相乘,sub_a的子矩阵大小为rm * bk ,sub_b的子矩阵大小为bk * rn
		,当前线程处理sub_a横方向上的第threadIdx.y个子块，处理sub_b竖方向上的第threadIdx.x个子块*/
		sub_matrix_multiply_0(result,sub_a,sub_b);
	}
	
	//将每个线程的计算结果写回全局内存
	//每个线程处理的子块，对应到原始C矩阵的坐标为[a_start_row + threadIdx.y * rm][b_start_col + threadIdx.x * rn]
	int save_c_row = a_start_row + threadIdx.y * RM;
	int save_c_col = b_start_col + threadIdx.x * RN;
	for(int i = 0; i < RM ; i++){
		for(int j = 0; j < RN ; j++){
			d_matrix_c[OFFSET(save_c_row + i,save_c_col + j,n)] = result[i][j];
		}
	}
}

__global__ void kernel_gemm2_1(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,int m,int n,int k){
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	
	//分配共享内存，用于加载a和b的各一小块子矩阵
	__shared__ float sub_a[BM][BK];
	__shared__ float sub_b[BK][BN];
	float result[RM][RN] = {0};//每个线程计算rm*rn个数据
	
	//TEST
	//if(blockIdx.x != 0 || blockIdx.y != 0 || idx != 5) return;
	//printf("idx = %d\n",idx);
								 
	/*将C矩阵m*n，切割成bm*bn的小块，每个block计算其中一小块*/
	//计算当前BLOCK要处理的a矩阵的首行号和尾行号
	int a_start_row = blockIdx.y * BM;//每个BLOCK处理bm行

	//计算当前BLOCK要处理的b矩阵的首列号和尾列号
	int b_start_col = blockIdx.x * BN;//每个BLOCK处理bn列
	
	/*无法将SUB_A = A[a_start_row:a_end_row][0 ：n]，SUB_B = B[0 : k][b_start_col:b_end_col]一次般入共享内存，因此对SUB_A进行横向上的切块，
	  对SUB_B进行竖向方向的切块。每次从SUB_A和SUB_B上取一个小块，然后计算这两个小块的矩阵乘法：A矩阵每块的大小bm*bk,B矩阵每块的大小为bk*bn*/
	//根据分块信息，计算总共需要进行的计算迭代数
	int block_compute_num = (k + BK - 1) / BK;
	
	//接下来考虑a[a_start_row:a_end_row][a_start_col:a_end_col],b[b_start_row:b_end_row][b_start_col:b_end_col]这两个小矩阵的加载方式
	//为了减少访存指令，每个线程load一个float4,也就是4个数据，计算每个线程需要Load的次数
	int block_load_num = blockDim.x * blockDim.y * 4;
	int load_a_gmem_num = (BM * BK + block_load_num - 1) / block_load_num;
	int load_b_gmem_num = (BK * BN + block_load_num - 1) / block_load_num;
	//printf("load_a_gmem_num = %d,load_b_gmem_num = %d\n",load_a_gmem_num,load_b_gmem_num);
	//依次加载一小块a和一小块b到共享内存中，然后启动计算
	for(int i = 0; i < block_compute_num; i++){
		//计算要加载的a矩阵的首列号和尾列号
		int a_start_col = i * BK;//每次迭代处理bk列
		
		//计算要加载的b矩阵的首行号和尾行号,等于a矩阵的首列号和尾列号
		int b_start_row = a_start_col;
#if 1		
		__syncthreads();//遗留问题：这里必须要有同步语句，否则会出现A子矩阵读取不正确问题，没有搞清楚什么原因
		//读取A的子矩阵
		for(int j = 0; j < load_a_gmem_num; j++){
			//当前线程读取的数据相对于sub_a子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_a[load_sub_a_row = offset / bk][load_sub_a_col = offset % bk] 
			//对应到原始a矩阵的坐标为a[a_start_row + min_a_row][a_start_col + min_a_col]
			int load_sub_a_row = (j * block_load_num + idx * 4) / BK;
			int load_sub_a_col = (j * block_load_num + idx * 4) % BK;
			if(load_sub_a_row < BM && load_sub_a_col < BK){
				//printf("%d,%d,%d,%d\n",load_sub_a_row,load_sub_a_col,a_start_row + load_sub_a_row,a_start_col + load_sub_a_col);
				FETCH_FLOAT4(sub_a[load_sub_a_row][load_sub_a_col]) = \
					FETCH_FLOAT4(d_matrix_a[OFFSET(a_start_row + load_sub_a_row,a_start_col + load_sub_a_col,k)]);
			}
		}
		//读取B的子矩阵
		for(int j = 0; j < load_b_gmem_num; j++){
			//当前线程读取的数据相对于sub_b子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_b[load_sub_b_row = offset / bn][load_sub_b_col = offset % bn] 
			//对应到原始b矩阵的坐标为b[b_start_row + min_b_row][b_start_col + min_b_col]
			int load_sub_b_row = (j * block_load_num + idx * 4) / BN;
			int load_sub_b_col = (j * block_load_num + idx * 4) % BN;
			if(load_sub_b_row < BK && load_sub_b_col < BN){
				FETCH_FLOAT4(sub_b[load_sub_b_row][load_sub_b_col]) = \
					FETCH_FLOAT4(d_matrix_b[OFFSET(b_start_row + load_sub_b_row,b_start_col + load_sub_b_col,n)]);
			}
		}
#endif
		__syncthreads();
		//加载完成后，启动计算，计算结果保存在寄存器中
		/*每个线程从sub_a和sub_b中再取一个子矩阵相乘,sub_a的子矩阵大小为rm * bk ,sub_b的子矩阵大小为bk * rn
		,当前线程处理sub_a横方向上的第threadIdx.y个子块，处理sub_b竖方向上的第threadIdx.x个子块*/
		sub_matrix_multiply_1(result,sub_a,sub_b);
	}
	
	//将每个线程的计算结果写回全局内存
	//每个线程处理的子块，对应到原始C矩阵的坐标为[a_start_row + threadIdx.y * rm][b_start_col + threadIdx.x * rn]
	int save_c_row = a_start_row + threadIdx.y * RM;
	int save_c_col = b_start_col + threadIdx.x * RN;
	for(int i = 0; i < RM ; i++){
		for(int j = 0; j < RN ; j++){
			d_matrix_c[OFFSET(save_c_row + i,save_c_col + j,n)] = result[i][j];
		}
	}
}

__global__ void kernel_gemm2_2(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,int m,int n,int k){
	int idx = threadIdx.y * blockDim.x + threadIdx.x;
	
	//分配共享内存，用于加载a和b的各一小块子矩阵
	__shared__ float sub_a[2][BM][BK];
	__shared__ float sub_b[2][BK][BN];
	float result[RM][RN] = {0};//每个线程计算rm*rn个数据
	
	//TEST
	//if(blockIdx.x != 0 || blockIdx.y != 0 || idx != 5) return;
	//printf("idx = %d\n",idx);
								 
	/*将C矩阵m*n，切割成bm*bn的小块，每个block计算其中一小块*/
	//计算当前BLOCK要处理的a矩阵的首行号和尾行号
	int a_start_row = blockIdx.y * BM;//每个BLOCK处理bm行

	//计算当前BLOCK要处理的b矩阵的首列号和尾列号
	int b_start_col = blockIdx.x * BN;//每个BLOCK处理bn列
	
	/*无法将SUB_A = A[a_start_row:a_end_row][0 ：n]，SUB_B = B[0 : k][b_start_col:b_end_col]一次般入共享内存，因此对SUB_A进行横向上的切块，
	  对SUB_B进行竖向方向的切块。每次从SUB_A和SUB_B上取一个小块，然后计算这两个小块的矩阵乘法：A矩阵每块的大小bm*bk,B矩阵每块的大小为bk*bn*/
	//根据分块信息，计算总共需要进行的计算迭代数
	int block_compute_num = (k + BK - 1) / BK;
	
	//接下来考虑a[a_start_row:a_end_row][a_start_col:a_end_col],b[b_start_row:b_end_row][b_start_col:b_end_col]这两个小矩阵的加载方式
	//为了减少访存指令，每个线程load一个float4,也就是4个数据，计算每个线程需要Load的次数
	int block_load_num = blockDim.x * blockDim.y * 4;
	int load_a_gmem_num = (BM * BK + block_load_num - 1) / block_load_num;
	int load_b_gmem_num = (BK * BN + block_load_num - 1) / block_load_num;
	//printf("load_a_gmem_num = %d,load_b_gmem_num = %d\n",load_a_gmem_num,load_b_gmem_num);
	/******先加载第一个缓冲区数据*****/
	int buf_indx = 0;
	//读取A的子矩阵
	for(int j = 0; j < load_a_gmem_num; j++){
		//当前线程读取的数据相对于sub_a子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_a[load_sub_a_row = offset / bk][load_sub_a_col = offset % bk] 
		//对应到原始a矩阵的坐标为a[a_start_row + min_a_row][a_start_col + min_a_col]
		int load_sub_a_row = (j * block_load_num + idx * 4) / BK;
		int load_sub_a_col = (j * block_load_num + idx * 4) % BK;
		if(load_sub_a_row < BM && load_sub_a_col < BK){
			//printf("%d,%d,%d,%d\n",load_sub_a_row,load_sub_a_col,a_start_row + load_sub_a_row,a_start_col + load_sub_a_col);
			FETCH_FLOAT4(sub_a[buf_indx][load_sub_a_row][load_sub_a_col]) = \
				FETCH_FLOAT4(d_matrix_a[OFFSET(a_start_row + load_sub_a_row,load_sub_a_col,k)]);
		}
	}
	//读取B的子矩阵
	for(int j = 0; j < load_b_gmem_num; j++){
		//当前线程读取的数据相对于sub_b子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_b[load_sub_b_row = offset / bn][load_sub_b_col = offset % bn] 
		//对应到原始b矩阵的坐标为b[b_start_row + min_b_row][b_start_col + min_b_col]
		int load_sub_b_row = (j * block_load_num + idx * 4) / BN;
		int load_sub_b_col = (j * block_load_num + idx * 4) % BN;
		if(load_sub_b_row < BK && load_sub_b_col < BN){
			FETCH_FLOAT4(sub_b[buf_indx][load_sub_b_row][load_sub_b_col]) = \
				FETCH_FLOAT4(d_matrix_b[OFFSET(load_sub_b_row,b_start_col + load_sub_b_col,n)]);
		}
	}
	__syncthreads();
	//在每轮迭代中，先加载第二个缓冲区数据，然后启动第一块缓冲区的计算，这样缓冲和计算可以并行，从而隐藏加载的时间
	//由于首块已经加载，所以迭代从1开始
	for(int i = 1; i < block_compute_num; i++){
		//计算要加载的a矩阵的首列号和尾列号
		int a_start_col = i * BK;//每次迭代处理bk列
		
		//计算要加载的b矩阵的首行号和尾行号,等于a矩阵的首列号和尾列号
		int b_start_row = a_start_col;
#if 1		
		//读取A的子矩阵
		for(int j = 0; j < load_a_gmem_num; j++){
			//当前线程读取的数据相对于sub_a子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_a[load_sub_a_row = offset / bk][load_sub_a_col = offset % bk] 
			//对应到原始a矩阵的坐标为a[a_start_row + min_a_row][a_start_col + min_a_col]
			int load_sub_a_row = (j * block_load_num + idx * 4) / BK;
			int load_sub_a_col = (j * block_load_num + idx * 4) % BK;
			if(load_sub_a_row < BM && load_sub_a_col < BK){
				//printf("%d,%d,%d,%d\n",load_sub_a_row,load_sub_a_col,a_start_row + load_sub_a_row,a_start_col + load_sub_a_col);
				FETCH_FLOAT4(sub_a[1 - buf_indx][load_sub_a_row][load_sub_a_col]) = \
					FETCH_FLOAT4(d_matrix_a[OFFSET(a_start_row + load_sub_a_row,a_start_col + load_sub_a_col,k)]);
			}
		}
		//读取B的子矩阵
		for(int j = 0; j < load_b_gmem_num; j++){
			//当前线程读取的数据相对于sub_b子矩阵的偏移offset = (j * block_load_num + idx * 4),对应的坐标为：sub_b[load_sub_b_row = offset / bn][load_sub_b_col = offset % bn] 
			//对应到原始b矩阵的坐标为b[b_start_row + min_b_row][b_start_col + min_b_col]
			int load_sub_b_row = (j * block_load_num + idx * 4) / BN;
			int load_sub_b_col = (j * block_load_num + idx * 4) % BN;
			if(load_sub_b_row < BK && load_sub_b_col < BN){
				FETCH_FLOAT4(sub_b[1 - buf_indx][load_sub_b_row][load_sub_b_col]) = \
					FETCH_FLOAT4(d_matrix_b[OFFSET(b_start_row + load_sub_b_row,b_start_col + load_sub_b_col,n)]);
			}
		}
#endif
		//此处不进行同步，直接启动对第一缓冲区数据的计算
		//__syncthreads();
		//加载完成后，启动计算，计算结果保存在寄存器中
		/*每个线程从sub_a和sub_b中再取一个子矩阵相乘,sub_a的子矩阵大小为rm * bk ,sub_b的子矩阵大小为bk * rn
		,当前线程处理sub_a横方向上的第threadIdx.y个子块，处理sub_b竖方向上的第threadIdx.x个子块*/
		sub_matrix_multiply_1(result,(float (*)[BK])(&sub_a[buf_indx][0][0]),(float (*)[BN])(&sub_b[buf_indx][0][0]));
		//此处同步，确保另一个缓冲区数据已经加载完成
		__syncthreads();
		//更换缓冲区
		buf_indx = 1 - buf_indx;
	}
	//完成最后一个缓冲区只加载没有计算，在这里完成计算
	sub_matrix_multiply_1(result,(float (*)[BK])(&sub_a[buf_indx][0][0]),(float (*)[BN])(&sub_b[buf_indx][0][0]));
	
	//将每个线程的计算结果写回全局内存
	//每个线程处理的子块，对应到原始C矩阵的坐标为[a_start_row + threadIdx.y * rm][b_start_col + threadIdx.x * rn]
	int save_c_row = a_start_row + threadIdx.y * RM;
	int save_c_col = b_start_col + threadIdx.x * RN;
	for(int i = 0; i < RM ; i++){
		for(int j = 0; j < RN ; j++){
			d_matrix_c[OFFSET(save_c_row + i,save_c_col + j,n)] = result[i][j];
		}
	}
}

void gemm_compute(float *d_matrix_a,float *d_matrix_b,float *d_matrix_c,float *h_matrix_c,float *h_matrix_c_gpu,int method,int bcheck){
	//计算一个BLOCK内的线程布局，每个线程处理RM*RN个数据
	int blockdim_x = (BN + RN - 1) / RN;
	int blockdim_y = (BM + RM - 1) / RM;
	dim3 block(blockdim_x,blockdim_y);
	//计算网格上的BLOCK数，每个BLOCK处理BM*BN个数据
	dim3 grid((N + BN - 1) / BN,(M + BM - 1) / BM);
	
	//避免上一方法计算结果的影响，先清0结果
	cudaMemset(d_matrix_c,0,sizeof(float) * M * N);
	printf("grid(%d,%d) block(%d,%d)\n",(N + BN - 1) / BN,(M + BM - 1) / BM,blockdim_x,blockdim_y);
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	int type = (method & 0xf0) >> 4;
	int opti = (method & 0x0f);
	if(1 == type){//支持任意大小的矩阵
		if(0 == opti){
			kernel_gemm1_0<<<grid,block>>>(d_matrix_a,d_matrix_b,d_matrix_c,M,N,K);
		}else if(1 == opti){
			kernel_gemm1_1<<<grid,block>>>(d_matrix_a,d_matrix_b,d_matrix_c,M,N,K);
		}
	}else{
		if(((M % BM)|(N % BN)|(K % BK)|(BM % RM)|(BN % RN)|(K % 4)|(N % 4)) != 0){
			printf("unsupport method[0x%x]!\n",method);
			//return;
		}else{
			if(0 == opti){
				kernel_gemm2_0<<<grid,block>>>(d_matrix_a,d_matrix_b,d_matrix_c,M,N,K);
			}else if(1 == opti){
				kernel_gemm2_1<<<grid,block>>>(d_matrix_a,d_matrix_b,d_matrix_c,M,N,K);
			}else if(2 == opti){
				kernel_gemm2_2<<<grid,block>>>(d_matrix_a,d_matrix_b,d_matrix_c,M,N,K);
			}
		}
	}
	CUDA_CHECK_RETURN(cudaGetLastError());
	cudaEventRecord(stop);
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	float elapsedTime = 0.0;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("method[0x%x] cudaProcElapsed = %f\n",method,elapsedTime / 1000);
	cudaMemcpy(h_matrix_c_gpu,d_matrix_c,sizeof(float) * M * N,cudaMemcpyDeviceToHost);
	if(bcheck){
		check(h_matrix_c,h_matrix_c_gpu,M,N);
	}
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

/*
执行结果：

*/
void gemm_test(){
	float *h_matrix_a = (float *)malloc(sizeof(float) * M * K);
	float *h_matrix_b = (float *)malloc(sizeof(float) * K * N);
	float *h_matrix_c = (float *)malloc(sizeof(float) * M * N);
	float *h_matrix_c_gpu = (float *)malloc(sizeof(float) * M * N);
	
	srand((unsigned)time(NULL));
	for(int i = 0; i < M; i++){
		for(int j = 0; j < K; j++){
			h_matrix_a[i * K + j] = (rand() / (double)RAND_MAX) * 10;
			//h_matrix_a[i * K + j] = 0.8;
		}
	}
	for(int i = 0; i < K; i++){
		for(int j = 0; j < N; j++){
			h_matrix_b[i * N + j] = (rand() / (double)RAND_MAX) * 10;
			//h_matrix_b[i * N + j] = 1;
		}
	}
#if 0
	double iStart = cpuSecond();
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			h_matrix_c[i * N + j] = 0;
			for(int k = 0; k < K; k++){
				h_matrix_c[i * N + j] += h_matrix_a[i * K + k] * h_matrix_b[k * N + j];
			}
		}
	}
	double iElaps = cpuSecond() - iStart;
	printf("cpuProcElapsed = %f,h_matrix_c[0] = %f\n",iElaps,h_matrix_c[0]);
#endif
	float *d_matrix_a = NULL,*d_matrix_b = NULL,*d_matrix_c = NULL;
	cudaMalloc(&d_matrix_a,sizeof(float) * M * K);
	cudaMalloc(&d_matrix_b,sizeof(float) * K * N);
	cudaMalloc(&d_matrix_c,sizeof(float) * M * N);
	cudaMemcpy(d_matrix_a,h_matrix_a,sizeof(float) * M * K,cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_b,h_matrix_b,sizeof(float) * K * N,cudaMemcpyHostToDevice);
	
	gemm_compute0(d_matrix_a,d_matrix_b,d_matrix_c,h_matrix_c,h_matrix_c_gpu,0);
	//以navie计算结果为校验标准
	cudaMemcpy(h_matrix_c,d_matrix_c,sizeof(float) * M * N,cudaMemcpyDeviceToHost);
	//0x1[x]支持任意大小的矩阵
	gemm_compute(d_matrix_a,d_matrix_b,d_matrix_c,h_matrix_c,h_matrix_c_gpu,0x10,1);
	gemm_compute(d_matrix_a,d_matrix_b,d_matrix_c,h_matrix_c,h_matrix_c_gpu,0x11,1);
	//0x2[x]要求矩阵能分成整块
	gemm_compute(d_matrix_a,d_matrix_b,d_matrix_c,h_matrix_c,h_matrix_c_gpu,0x20,1);
	gemm_compute(d_matrix_a,d_matrix_b,d_matrix_c,h_matrix_c,h_matrix_c_gpu,0x21,1);
	gemm_compute(d_matrix_a,d_matrix_b,d_matrix_c,h_matrix_c,h_matrix_c_gpu,0x22,1);

	free(h_matrix_a);
	free(h_matrix_b);
	free(h_matrix_c);
	free(h_matrix_c_gpu);
	cudaFree(d_matrix_a);
	cudaFree(d_matrix_b);
	cudaFree(d_matrix_c);
}
