/*
 * chapter6.c
 *
 *  Created on: 2023Äê12ÔÂ14ÈÕ
 *      Author: zhongzhw
 */
#include "func_def.h"

static void test_1(){
	int a = 1,b = 2,c = 3;
	printf("addr(a) = %p,addr(b) = %p,addr(c) = %p\n",&a,&b,&c);

#pragma omp parallel shared(a) private(b) firstprivate(c)
	{
		int id = omp_get_thread_num();
		if(0 == id){
			printf("a = %d,addr(a) = %p\n",a,&a);
			printf("b = %d,addr(b) = %p\n",b,&b);
			printf("c = %d,addr(c) = %p\n",c,&c);
		}
	}
}

void chapter6_main(){
	test_1();
}

