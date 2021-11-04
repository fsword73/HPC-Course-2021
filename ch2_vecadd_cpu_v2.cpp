#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>

#define ARR_SIZE 1024*1024*256
int main(int argc,char* argv[]) {

    float* a = new float[ARR_SIZE];
    float* b = new float[ARR_SIZE];
    float* c = new float[ARR_SIZE];

    for(int i = 0; i < ARR_SIZE; i++) {
        a[i] = (float)(i & 0x3f);
        b[i] = (float)(~i & 0x3f);
        c[i] = 0;
    }
#pragma unroll(32)
    for(int i = 0; i < ARR_SIZE; i++) {
        c[i] = a[i] + b[i];
    }

    delete[] a;
    delete[] b;
    delete[] c;
    return 0; 
}