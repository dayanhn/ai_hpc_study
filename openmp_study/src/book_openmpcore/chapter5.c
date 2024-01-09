/*
 * chapter5.c
 *
 *  Created on: 2023��12��14��
 *      Author: zhongzhw
 */
#include "func_def.h"

/**
 1�����ʽ���й�����ѭ������
 */
static void test_parallel_for(){
	int num = 100;
#pragma omp parallel for
	for(int i = 0; i < num; i++){
		printf("i = %d,core num = %d\r\n",i,omp_get_thread_num());
	}
}

/**
 2����Լ
 ��Ӧ���Ӿ�Ϊ�� reduction (op:list)
 op��һ�������ı��������
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
