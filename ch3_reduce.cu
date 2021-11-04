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



#define NUM   4096

#define THREADS_PER_BLOCK_X  1024
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define ARR_SIZE   (THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y*THREADS_PER_BLOCK_Z)


//local block size,  (256, 1)
//total threads (H * W, 1)


//mask must 0
__global__ void
test_kernel(int*  buf, int*  res)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

	__shared__ int shared_Data[4096];
	shared_Data[tx] = buf[x];
    __syncthreads();

    for(int s=128; s>0; s = s>>1){
        if(tx < s){
            shared_Data[tx] = shared_Data[tx] + shared_Data[tx+s];
        }
        __syncthreads();
    }

	if (tx == 0){
		res[x] =  shared_Data[0];
	}
}


using namespace std;


int main() {

   // Size of vectors
    int n = ARR_SIZE;
 
    // Host vectors
    int *h_a;
    int *h_b;
 
    // Device input vectors
    int *d_a;
    int *d_b;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);
    size_t bytes_b = n*sizeof(int)/THREADS_PER_BLOCK_X;
 
    // Allocate memory  on host
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes_b);
 
    // Allocate memory  on GPU
    gpuErrchk(cudaMalloc(&d_a, bytes));
    gpuErrchk(cudaMalloc(&d_b, bytes_b));
 
    int i;
    // Initialize on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = i;
    }
    for( i = 0; i < n/1024; i++ ) {
        h_b[i] = 0;
    }
 
    // Copy from host to device
    gpuErrchk(cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice));
 
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
    test_kernel<<<gridSize, blockSize>>>(d_a,d_b);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("Cold Run costs  %f(ms)\n",time_elapsed);

    // Execute the kernel
    cudaEventRecord( start,0);
    test_kernel<<<gridSize, blockSize>>>(d_a,d_b);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("Warm Run costs  %f(ms)\n",time_elapsed);


    // Copy result back to host
    cudaMemcpy( h_b, d_b, bytes_b, cudaMemcpyDeviceToHost );
    printf("result %d\n", h_b[0]);

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
 
    // free host memory
    free(h_a);
    free(h_b);
 
    return 0;
}

