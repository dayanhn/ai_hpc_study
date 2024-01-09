/*
 * chapter10.c
 *
 *  Created on: 2023Äê12ÔÂ14ÈÕ
 *      Author: zhongzhw
 */
#include "func_def.h"
#include <omp.h>
#include <malloc.h>
#include <string.h>
#define N (1024*1024*100)
#define _OMP_
static void test_1(){
	float *a,*b,*c,*d;
	int i;
	a = (float *)malloc(N * sizeof(float));
	b = (float *)malloc(N * sizeof(float));
	c = (float *)malloc(N * sizeof(float));
	d = (float *)malloc(N * sizeof(float));
	memset(c,0,N * sizeof(float));
	memset(d,0,N * sizeof(float));

#ifdef _OMP_
#pragma omp target data map(to : a[0:N],b[0:N],c[0:N]) map(tofrom : d[0:N])
#endif
	{
#ifdef _OMP_
#pragma omp target
#pragma omp teams distribute parallel for simd
#endif
		for(i = 0; i < N; i++){
			a[i] = 2;
			b[i] = 3;
		}

#ifdef _OMP_
#pragma omp target
#pragma omp teams distribute parallel for simd
#endif
		for(i = 0; i < N; i++)
			c[i] = a[i] * b[i];

#ifdef _OMP_
#pragma omp target
#pragma omp teams distribute parallel for simd
#endif
		for( i = 0; i < N; i++)
			d[i] = a[i] + c[i];
	}

	printf("d[0] = %f\n",d[0]);
}

void chapter12_main(){
	test_1();
}

