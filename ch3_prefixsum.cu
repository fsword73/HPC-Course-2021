#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include<cuda_runtime.h>



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



#define NUM   1024

#define THREADS_PER_BLOCK_X  1024
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define ARR_SIZE   (THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y*THREADS_PER_BLOCK_Z)


//local block size,  (256, 1)
//total threads (H * W, 1)


//mask must 0
__global__ void
test_kernel(int* buf)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

	__shared__ int shared_Data[1024];
	shared_Data[tx] = tx&1;
    __syncthreads();

    for(int s=1; s<1024; s = s<<1){
        if( (tx & s) == s ){
            int data = shared_Data[s-1];
            shared_Data[tx] = shared_Data[tx] + data;
        }
        __syncthreads();
    }

	buf[x] =  shared_Data[tx];
}


using namespace std;


int main() {

   // Size of vectors
    int n = ARR_SIZE;
 
    // Host vectors
    int *h_a;
 
    // Device input vectors
    int *d_a;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);
 
    // Allocate memory  on host
    h_a = (int*)malloc(bytes);
 
    // Allocate memory  on GPU
    gpuErrchk(cudaMalloc(&d_a, bytes));
 
    int i;
    // Initialize on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = i;
    }

    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // total blocks in grid
    gridSize = (int)ceil((int)n/blockSize);
 
    //Profile GPU time
    float time_elapsed=0;
    cudaEvent_t start,stop;

    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    
    // Execute the kernel
    cudaEventRecord( start,0);
    test_kernel<<<gridSize, blockSize>>>(d_a);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("Cold Run costs  %f(ms)\n",time_elapsed);

    // Execute the kernel
    cudaEventRecord( start,0);
    test_kernel<<<gridSize, blockSize>>>(d_a);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("Warm Run costs  %f(ms)\n",time_elapsed);


    // Copy result back to host
    cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost );
    for(int i=0;i < n; i+=32){
        for(int j=0; j<32; j++){ 
            printf("%4d ", h_a[i+j]);            
        }
        printf("\n");
    }

    // free device memory
    cudaFree(d_a);
 
    // free host memory
    free(h_a);
 
    return 0;
}

