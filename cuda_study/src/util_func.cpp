#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <driver_types.h>


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void check(float *a,float *b,int m,int n){
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			if(fabs(a[i * n + j] - b[i * n + j]) > 0.1){
				printf("check error,a[%d][%d] = %f,b[%d][%d] = %f!\n",i,j,a[i * n + j],i,j,b[i * n + j]);
				return;
			}
		}
	}
	printf("check ok!\n");
}

using namespace std;

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
