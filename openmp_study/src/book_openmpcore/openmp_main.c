/*
 ============================================================================
 Name        : openmp_test.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include "func_def.h"

void get_core_num(){
#pragma omp parallel
    {
        printf("core num = %d\r\n",omp_get_num_threads());
    }
}

void test_1(){
#pragma omp parallel
	{
		int id = omp_get_thread_num();
		printf("id = %d\n",id);
	}
}


void test(){
#pragma omp parallel num_threads(2)
	{
		int id = omp_get_thread_num();
		printf("id = %d\n",id);
		test_1();
	}
}


int openmp_main(){
	//get_cpu_num();
	//chapter5_main();
	//chapter6_main();
	//chapter7_main();
	//chapter10_main();
	chapter12_main();
	//test();
    return 0;
}
