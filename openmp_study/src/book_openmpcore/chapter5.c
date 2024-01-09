/*
 * chapter5.c
 *
 *  Created on: 2023年12月14日
 *      Author: zhongzhw
 */
#include "func_def.h"

/**
 1、组合式并行共享工作循环构造
 */
static void test_parallel_for(){
	int num = 100;
#pragma omp parallel for
	for(int i = 0; i < num; i++){
		printf("i = %d,core num = %d\r\n",i,omp_get_thread_num());
	}
}

/**
 2、归约
 对应的子句为： reduction (op:list)
 op是一个基本的标量运算符
 */

static void test_reduction(){
	int num = 100;
	int A[100];
	double ave = 0;
	init_vector(A, num, 2);
#pragma omp parallel for reduction (+:ave)
	for(int i = 0; i < num; i++){
		ave += A[i];
	}
	ave = ave / num;
	printf("ave = %f\n",ave);
}

static void test_schedule(){
	int num = 10000;
	int A[10000];
	double ave = 0;
	init_vector(A, num, 2);
#pragma omp parallel for reduction (+:ave) schedule(static,8)
	for(int i = 0; i < num; i++){
		ave += A[i];
	}
	ave = ave / num;
	printf("ave = %f\n",ave);
}

static void test_1(){
	#pragma omp parallel
	{
		int id = omp_get_thread_num();

		#pragma omp for schedule(static,1)
		for(int i = 0; i < 20; i++){
			int l_id = omp_get_thread_num();
			printf("i = %d,id = %d,l_id = %d\n",i,id,l_id);
		}
	}
}

void chapter5_main(){
	//test_parallel_for();
	//test_reduction();
	//test_schedule();
	test_1();
}
