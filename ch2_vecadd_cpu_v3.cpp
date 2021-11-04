#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include <sys/time.h>
using namespace std;

#define ARR_SIZE 1024*1024*256


void test_plain(float* a, float* b, float* c, long n)
{
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    for(long i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec)*1000000 + (double)(t2.tv_usec - t1.tv_usec);

    cout<<"No optimziation time = "<< timeuse << "  microsecond" << endl;  

}


 void inline test_cacheline_inline(float* a, float* b, float* c, long n, long interval)
{
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    for(long j = 0; j < interval ; j++){
        for(long i= 0; i < n; i+=interval) {
            c[i+j] = a[i+j] + b[i+j];
        }
    }
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec)*1000000 + (double)(t2.tv_usec - t1.tv_usec);

    cout<<"Interleaved Distance[" << interval << "] time= "<< timeuse << "  microsecond" << endl;  
}


static void test_cacheline(float* a, float* b, float* c, long n, long interval)
{
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    for(long j = 0; j < interval ; j++){
        for(long i= 0; i < n; i+=interval) {
            c[i+j] = a[i+j] + b[i+j];
        }
    }
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec)*1000000 + (double)(t2.tv_usec - t1.tv_usec);

    cout<<"Interleaved Distance[" << interval << "] time= "<< timeuse << "  microsecond" << endl;  
}


void test_unroll(float* a, float* b, float* c, long n, long unroll_size)
{
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    for(long i = 0; i < n; i+= unroll_size ){
        for(long j= 0; j < unroll_size; j++) {
            c[i+j] = a[i+j] + b[i+j];
        }
    }
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec)*1000000 + (double)(t2.tv_usec - t1.tv_usec);

    cout<<"Loop Unroll Size[" << unroll_size <<"] time= "<< timeuse << "  microsecond" << endl;  
}


int main(int argc,char* argv[]) {

    float* a = new float[ARR_SIZE];
    float* b = new float[ARR_SIZE];
    float* c = new float[ARR_SIZE];

    for(int i = 0; i < ARR_SIZE; i++) {
        a[i] = (float)(i & 0x3f);
        b[i] = (float)(~i & 0x3f);
        c[i] = 0;
    }

    long n = ARR_SIZE;
    test_plain(a,b,c,n);

    for(int i=2; i < 1024; i = i* 2)
    {
        test_cacheline_inline(a,b,c,n,i);
    }

    int i0;
    i0 = max(argc*argc,argc+1); 

    for(int i=i0; i < 1024; i = i* 2)
    {
        test_cacheline(a,b,c,n,i);
    }

    for(int i=i0; i < 1024; i = i* 2)
    {
        test_unroll(a,b,c,n,i);
    }


    delete[] a;
    delete[] b;
    delete[] c;
    return 0; 
}