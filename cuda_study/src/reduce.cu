#include "util_func.h"

#define THREAD_PER_BLOCK 1024
#define WARP_SIZE 32
#define N (160000000)
//#define N (2049)

/*
reduce方法0
处理流程：
1、Block内的每个线程从全局内存中读取一个数到共享内存中
2、第1轮迭代，由序号为偶数的线程处理相邻两个数的相加，如0号线程累加0和1号数据，1号线程空闲，2号线程累加2号和3号数据，3号线程空闲。。。。
2、第2轮迭代，参与计算的线程数减半，0号线程累加0号和2号数据，1,2,3号线程空闲，4号线程累加4和6号数据。。。。
3、反复迭代，直至都累加到0号数据

*/
__global__ void kernel_reduce0(float *d_in,float *d_out,int data_num){
	__shared__ float sdata[THREAD_PER_BLOCK];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	if(i < data_num){
		sdata[tid] = d_in[i];
	}else{
		sdata[tid] = 0;
	}
	__syncthreads();
	//处理步长按2的幕递增
	for(int s = 1; s < blockDim.x; s *= 2){
		//只有线程号为步长的偶数倍时才处理数据
		if((tid % (2*s)) == 0){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if(tid == 0) d_out[blockIdx.x] = sdata[tid];
}

/*
优化1：解决warp divergence
kernel_reduce0存在的问题：
在一个warp内，第一轮迭代有一半的线程只加载了数据，没有参与计算，第二轮迭代，只有1/4的线程参与了计算，越往后真正工作的线程数越少，因此造成了计算资源的浪费。
问题原因：线程与处理的数据的映射关系不合理，每个线程只处理对应位置的数据，造成线程的使用不连续
解决方法：每次都只使用BLOCK内前面的连续线程处理数据，比如：
第1轮迭代，0号线程处理0，1号数据，1号线程处理2，3号数据，2号线程处理4，5号数据。这样一个BLOCK内后半部分的线程可以提前结束，不需要再调度
第2轮迭代，0号线程处理0，2号数据，1号数据处理4，6号线程。。。。这样又少一半的线程可以提前结束，不需要再调度

*/
__global__ void kernel_reduce1(float *d_in,float *d_out,int data_num){
	__shared__ float sdata[THREAD_PER_BLOCK];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	if(i < data_num){
		sdata[tid] = d_in[i];
	}else{
		sdata[tid] = 0;
	}
	__syncthreads();
	//处理步长按2的幕递增
	for(int s = 1; s < blockDim.x; s *= 2){
		//使用连续的线程处理数据，因为当前步长为s，则线程要处理的第一个位置是 tid * 2 *s，第二个数据数据是tid * 2 *s + s
		int index = tid * 2 * s;
		if((index + s) < blockDim.x){
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	if(tid == 0) d_out[blockIdx.x] = sdata[tid];
}

/*
优化2：解决存储体冲突
kernel_reduce1存在的问题：
首先理解存储体的概念：共享内存被分为32个同样大小的内存体(banks，对应线程束中32个线程)。 共享内存是一个可以被同时访问的一维地址空间。 如果线程束32个线程通过共享内存32个banks加载或存储数据，且每个bank上只访问不多于一个的内存地址，则可由一个内存事务来完成。 否则，必然
会产生存储体冲突，需要由多个内存事务来完成，以致降低了内存带宽的利用率。
在reduce1的处理中，Warp中的每个线程会访问两个数据，如第1轮迭代中0号线程访问0，1号数据，16号线程访问32，33号数据，因此0到15号线程与16到31号线程的内存访问都会产生冲突。
解决方法：如果让warp中的每个线程同时访问的数据位置都是连续的，比如依次是0到31号数据就不会产生冲突。因此步长不能是从2开始，而要大于32开始，这样我们可以将步长从blockDim.x/2开始，
每次折半，直到步长为1》这样在第1轮迭代中，0号线程先后访问0，512号数据，1号线程先后访问1，512号数据。。。。在一个warp中的线程在每次访问数据位置都是连续的，因此不会产生冲突

*/
__global__ void kernel_reduce2(float *d_in,float *d_out,int data_num){
	__shared__ float sdata[THREAD_PER_BLOCK];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	if(i < data_num){
		sdata[tid] = d_in[i];
	}else{
		sdata[tid] = 0;
	}
	__syncthreads();
	//处理步长按2的折半递减
	for(int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if(tid == 0) d_out[blockIdx.x] = sdata[tid];
}

/*
优化3：让工作量少的线程尽可能多干活
kernel_reduce2存在的问题：
当THREAD_PER_BLOCK为1024时，后512个线程只加载了数据，并没有执行计算。因为加载的数据是一定的，因此可以尽可能的利用这些工作量少的线程干一部分计算的活，从而减少其它线程的工作量
解决方法：
一个block内的线程负责两个block长度的数据计算，即第i号block中的线程在加载数据时，需要加载第i*2和i*2+1号block中的数据，并相加存到共享内存中，这样block数可以减半
*/
__global__ void kernel_reduce3(float *d_in,float *d_out,int data_num){
	__shared__ float sdata[THREAD_PER_BLOCK];

	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int tid = threadIdx.x;
	if((i + blockDim.x) < data_num){
		sdata[tid] = d_in[i] + d_in[i + blockDim.x];
	}else if(i < data_num){
		sdata[tid] = d_in[i];
	}else{
		sdata[tid] = 0;
	}
	__syncthreads();
	//处理步长按2的折半递减
	for(int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if(tid == 0) d_out[blockIdx.x] = sdata[tid];
}

/*
优化4：当在一个block内只剩一个warp时，采用warp的shuffle指令，避免共享内存的访问，同时也避免使用同步语句
kernel_reduce3存在的问题：
当block内只剩一个warp时，依然使用了同步语句__syncthreads，是不必须的
解决方法：
当只剩最后一个warp时，采用shfl指令进行累加，shfl指令是线程同步的，然后也避免了共享内存的访问
*/
__global__ void kernel_reduce4(float *d_in,float *d_out,int data_num){
	__shared__ float sdata[THREAD_PER_BLOCK];

	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int tid = threadIdx.x;
	float sum = 0;
	if((i + blockDim.x) < data_num){
		sdata[tid] = d_in[i] + d_in[i + blockDim.x];
	}else if(i < data_num){
		sdata[tid] = d_in[i];
	}else{
		sdata[tid] = 0;
	}
	__syncthreads();
	//处理步长按2的折半递减
	for(int s = blockDim.x / 2; s >= 32; s >>= 1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
		}	
		__syncthreads();
	}
	//最后一个warp将共享内存的数据存入寄存器，通过warp内的相互寄存器访问实现累加
	if(tid < 32) {
		sum = sdata[tid];
		__syncthreads();
	}
	//前16个线程累加后16个线程的sum
	sum += __shfl_down_sync(0xffffffff,sum,16);
	sum += __shfl_down_sync(0xffffffff,sum,8);
	sum += __shfl_down_sync(0xffffffff,sum,4);
	sum += __shfl_down_sync(0xffffffff,sum,2);
	//第0号线程累加第1号线程的sum，最后得到结果
	sum += __shfl_down_sync(0xffffffff,sum,1);
	if(tid == 0) d_out[blockIdx.x] = sum;
}

/*
优化5：循环展开，不使用for语句
*/
__global__ void kernel_reduce5(float *d_in,float *d_out,int data_num){
	__shared__ float sdata[THREAD_PER_BLOCK];

	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int tid = threadIdx.x;
	float sum = 0;
	if((i + blockDim.x) < data_num){
		sdata[tid] = d_in[i] + d_in[i + blockDim.x];
	}else if(i < data_num){
		sdata[tid] = d_in[i];
	}else{
		sdata[tid] = 0;
	}
	__syncthreads();
	//前512个线程处理
	if(THREAD_PER_BLOCK > 512){
		if(tid < 512){
			sdata[tid] += sdata[tid + 512];
		}	
		__syncthreads();
	}
	//前256个线程处理
	if(THREAD_PER_BLOCK > 256){
		if(tid < 256){
			sdata[tid] += sdata[tid + 256];
		}	
		__syncthreads();
	}
	//前128个线程处理
	if(THREAD_PER_BLOCK > 128){
		if(tid < 128){
			sdata[tid] += sdata[tid + 128];
		}	
		__syncthreads();
	}
	//前64个线程处理
	if(THREAD_PER_BLOCK > 64){
		if(tid < 64){
			sdata[tid] += sdata[tid + 64];
		}	
		__syncthreads();
	}
	//前32个线程处理
	if(THREAD_PER_BLOCK > 32){
		if(tid < 32){
			sdata[tid] += sdata[tid + 32];
		}	
		__syncthreads();
	}
	
	//最后一个warp将共享内存的数据存入寄存器，通过warp内的相互寄存器访问实现累加
	if(tid < 32) {
		sum = sdata[tid];
		__syncthreads();
	}
	//前16个线程累加后16个线程的sum
	sum += __shfl_down_sync(0xffffffff,sum,16);
	sum += __shfl_down_sync(0xffffffff,sum,8);
	sum += __shfl_down_sync(0xffffffff,sum,4);
	sum += __shfl_down_sync(0xffffffff,sum,2);
	//第0号线程累加第1号线程的sum，最后得到结果
	sum += __shfl_down_sync(0xffffffff,sum,1);
	if(tid == 0) d_out[blockIdx.x] = sum;
}

/*
优化6：在优化3中，每个block线程处理两个block的数据，执行性能有了很大的提升，这里可以考虑一个block线程处理3，4。。。。个block数据，这样启动的BLOCK就会减少，
可以实测得到一个最优的配置，这里不再展开，处理方式类似于优化3
*/

/*
优化7：在前面的优化中，只有最后一个warp时才采用shfl指令，可以修改为每个warp都采用shfl指令。
如果THREAD_PER_BLOCK是1024，刚好可以分成32个warp，每个warp累加32个数据，得到32个结果，然后最后一个warp累加这32个数据
*/
__global__ void kernel_reduce7(float *d_in,float *d_out,int data_num){
	__shared__ float sdata[WARP_SIZE];

	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int tid = threadIdx.x;
	float sum = 0;
	//从全局内存中加载数据到寄存器中，同一个warp通过寄存器进行累加
	if((i + blockDim.x) < data_num){
		sum = d_in[i] + d_in[i + blockDim.x];
	}else if(i < data_num){
		sum = d_in[i];
	}else{
		sum = 0;
	}
	__syncthreads();
	
	//每个warp的前16个线程累加后16个线程的sum
	sum += __shfl_down_sync(0xffffffff,sum,16);
	sum += __shfl_down_sync(0xffffffff,sum,8);
	sum += __shfl_down_sync(0xffffffff,sum,4);
	sum += __shfl_down_sync(0xffffffff,sum,2);
	//第0号线程累加第1号线程的sum，最后得到结果
	sum += __shfl_down_sync(0xffffffff,sum,1);
	
	int lane_id = tid % WARP_SIZE;
	int warp_id = tid / WARP_SIZE;
	//每个warp的第0个线程将自己累加的值存入共享内存中
	if(0 == lane_id)
		sdata[warp_id] = sum;
	__syncthreads();
	
	
	//最后一个warp将共享内存的数据存入寄存器，通过warp内的相互寄存器访问实现累加
	if(tid < 32) {
		sum = sdata[tid];
		__syncthreads();
	}
	//前16个线程累加后16个线程的sum
	sum += __shfl_down_sync(0xffffffff,sum,16);
	sum += __shfl_down_sync(0xffffffff,sum,8);
	sum += __shfl_down_sync(0xffffffff,sum,4);
	sum += __shfl_down_sync(0xffffffff,sum,2);
	//第0号线程累加第1号线程的sum，最后得到结果
	sum += __shfl_down_sync(0xffffffff,sum,1);
	if(tid == 0) d_out[blockIdx.x] = sum;
}

void reduce_method_test(float *p_dev_in,float *p_dev_out1,float *p_dev_out2,int block_num,int method){
	//优化6没有实现
	if(6 == method) return;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	int data_num = N;
	float *d_in = p_dev_in;
	float *d_out = p_dev_out1;
	/*循环计算，直到最后只用一个BLOCK，每个block会将该block负责的数据累加成一个值，最后存放在d_out中*/
	while(block_num >= 1){
		if(1 == method){
			kernel_reduce1<<<block_num,THREAD_PER_BLOCK>>>(d_in,d_out,data_num);
		}else if(2 == method){
			kernel_reduce2<<<block_num,THREAD_PER_BLOCK>>>(d_in,d_out,data_num);
		}else if(3 == method){
			block_num = (block_num + 1) / 2;
			//如果只有一个BLOCK的数据，则也要启动一个BLOCK进行计算
			if(0 == block_num) block_num = 1;
			kernel_reduce3<<<block_num,THREAD_PER_BLOCK>>>(d_in,d_out,data_num);
		}else if(4 == method){
			block_num = (block_num + 1) / 2;
			//如果只有一个BLOCK的数据，则也要启动一个BLOCK进行计算
			if(0 == block_num) block_num = 1;
			kernel_reduce4<<<block_num,THREAD_PER_BLOCK>>>(d_in,d_out,data_num);
		}else if(5 == method){
			block_num = (block_num + 1) / 2;
			//如果只有一个BLOCK的数据，则也要启动一个BLOCK进行计算
			if(0 == block_num) block_num = 1;
			kernel_reduce5<<<block_num,THREAD_PER_BLOCK>>>(d_in,d_out,data_num);
		}else if(7 == method){
			block_num = (block_num + 1) / 2;
			//如果只有一个BLOCK的数据，则也要启动一个BLOCK进行计算
			if(0 == block_num) block_num = 1;
			kernel_reduce7<<<block_num,THREAD_PER_BLOCK>>>(d_in,d_out,data_num);
		}else{
			kernel_reduce0<<<block_num,THREAD_PER_BLOCK>>>(d_in,d_out,data_num);
		}
		//下次需要处理的数据量为当前的总block数
		data_num = block_num;
		//如果最后累加成一个数据则结束计算
		if(data_num == 1){
			break;
		}
		//计算下一次核函数需要启动的block数
		block_num = (data_num + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
		//将当前的输出数据作为下一次计算的输入
		d_in = d_out;
		//使用另一块输出数据BUF
		if(d_out == p_dev_out1){
			d_out = p_dev_out2;
		}else{
			d_out = p_dev_out1;
		}
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime = 0.0;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	
	float result = 0;
	cudaMemcpy(&result,d_out,sizeof(float),cudaMemcpyDeviceToHost);
	printf("reuduce%d cudaProcElapsed = %f,result = %f\n",method,elapsedTime / 1000,result);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

/*
执行结果：
cpuProcElapsed = 0.185327,result = 79999392.708566
reuduce0 cudaProcElapsed = 0.007329,result = 79999392.000000
reuduce1 cudaProcElapsed = 0.004701,result = 79999392.000000
reuduce2 cudaProcElapsed = 0.004509,result = 79999392.000000
reuduce3 cudaProcElapsed = 0.002395,result = 79999392.000000
reuduce4 cudaProcElapsed = 0.001909,result = 79999392.000000
reuduce5 cudaProcElapsed = 0.001788,result = 79999392.000000
reuduce7 cudaProcElapsed = 0.001322,result = 79999392.000000
*/
void reduce_test(){
	float *p_host_in = (float *)malloc(sizeof(float) * N);
	srand((unsigned)time(NULL));
	for(int i = 0; i < N; i++){
		p_host_in[i] = rand() /(double)RAND_MAX;
		//p_host_in[i] = 1.0;
	}
	double iStart = cpuSecond();
	double result = 0;
	for(int i = 0; i < N; i++){
		result +=  p_host_in[i];
	}
	double iElaps = cpuSecond() - iStart;
	printf("cpuProcElapsed = %f,result = %f\n",iElaps,result);

	cudaSetDevice(0);
	
	int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
	float *p_dev_in = NULL;
	//p_dev_out1和p_dev_out2交替使用，如果本次p_dev_out1做为输出，则下一次p_dev_out2做为输出，p_dev_out1作为输入
	float *p_dev_out1 = NULL;
	float *p_dev_out2 = NULL;
	cudaMalloc(&p_dev_in,sizeof(float) * N);
	cudaMalloc(&p_dev_out1,sizeof(float) * block_num);
	cudaMalloc(&p_dev_out2,sizeof(float) * block_num);
	cudaMemcpy(p_dev_in,p_host_in,sizeof(float) * N,cudaMemcpyHostToDevice);
	for(int i = 0; i < 8; i++){
		reduce_method_test(p_dev_in,p_dev_out1,p_dev_out2,block_num,i);
	}
	cudaFree(p_dev_in);
	cudaFree(p_dev_out1);
	cudaFree(p_dev_out2);
	free(p_host_in);
}
