/*
 * util_func.h
 *
 *  Created on: 2024年1月4日
 *      Author: zhongzhw
 */

#ifndef UTIL_FUNC_H_
#define UTIL_FUNC_H_
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>

using namespace std;

double cpuSecond();
void check(float *a,float *b,int m,int n);

void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#endif /* UTIL_FUNC_H_ */
