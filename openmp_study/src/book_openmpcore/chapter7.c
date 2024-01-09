/*
 * chapter7.c
 *
 *  Created on: 2023Äê12ÔÂ14ÈÕ
 *      Author: zhongzhw
 */
#include "func_def.h"
void sub_task_0(){
	int id = omp_get_thread_num();
	printf("sub_task_0,id = %d\n",id);
}

void sub_task_1(){
	int id = omp_get_thread_num();
	printf("sub_task_1,id = %d\n",id);
}

void sub_task_2(){
	int id = omp_get_thread_num();
	printf("sub_task_2,id = %d\n",id);
}

void sub_task_3(){
	int id = omp_get_thread_num();
	printf("sub_task_3,id = %d\n",id);
}


void task_0(){
	int id = omp_get_thread_num();
	printf("task_0,id = %d\n",id);
#if 0
#pragma omp parallel
	{
#pragma omp single
		{
#pragma omp task
			{
				sub_task_0();
			}
#pragma omp task
			{
				sub_task_1();
			}
#pragma omp task
			{
				sub_task_2();
			}
#pragma omp task
			{
				sub_task_3();
			}
		}
	}
#endif
}

void task_1(){
	int id = omp_get_thread_num();
	printf("task_1,id = %d\n",id);
}

void task_2(){
	int id = omp_get_thread_num();
	printf("task_2,id = %d\n",id);
}

void task_3(){
	int id = omp_get_thread_num();
	printf("task_3,id = %d\n",id);
}


static void test_1(){
#pragma omp parallel
	{
//#pragma omp single
		{
			int id = omp_get_thread_num();
			printf("create_task,id = %d\n",id);
#pragma omp task
			{
				task_0();
			}
#pragma omp task
			{
				task_1();
			}
#pragma omp task
			{
				task_2();
			}
#pragma omp task
			{
				task_3();
			}
		}
	}
}

static int fib(int n){
	int id = omp_get_thread_num();
	printf("n = %d,id = %d\n",n,id);
	int x,y;
	if(n < 2) return n;
#pragma omp task shared(x)
	x = fib(n - 1);
#pragma omp task shared(y)
	y = fib(n - 2);
#pragma omp taskwait
	return (x + y);
}

static void test_2(){
	int NW = 50;
	int re = 0;
#pragma omp parallel
	{
#pragma omp single
		{
			re = fib(NW);
		}
	}
	printf("re = %d\n",re);
}

void chapter7_main(){
	//test_1();
	test_2();
}

