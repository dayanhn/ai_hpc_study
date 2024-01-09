/*
 * chapter10.c
 *
 *  Created on: 2023Äê12ÔÂ14ÈÕ
 *      Author: zhongzhw
 */
#include "func_def.h"

static void test_1(){
#pragma omp parallel
	{
#pragma omp for collapse(2)
		for(int i = 0; i < 2; i++){
			for(int j = 0 ; j < 4; j++){
				int id = omp_get_thread_num();
				printf("id = %d\n",id);
			}
		}
	}
}

void chapter10_main(){
	test_1();
}

