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

#define THREADS_PER_BLOCK_X  1
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define ARR_SIZE   (THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y*THREADS_PER_BLOCK_Z)
#define OUTER_LOOPS        10000
#define INNER_LOOPS        100



//local block size,  (256, 1)
//total threads (H * W, 1)


//mask must 0
__global__ void
test_kernel(int* __restrict__ buf, int mask, int outerLloops)
{

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int shared_Data[4096];
	shared_Data[threadIdx.x] = buf[x];


	int temp = threadIdx.x;
	for(int i = 0; i < outerLloops; i++)	{
		for(int j = 0; j < INNER_LOOPS; j++)
		{
            //Compiler never to optimize it by mask = 0;
			temp = shared_Data[temp] & mask;
            
		}
	}
	if (temp > 0)
	{
		buf[x] = temp;
	}
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
 
    // Copy from host to device
    gpuErrchk(cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice));
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1;
 
    // total blocks in grid
    gridSize = (int)ceil((int)n/blockSize);
 
    //Profile GPU time
    float time_elapsed=0;
    cudaEvent_t start,stop;

    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    
    // Execute the kernel
    cudaEventRecord( start,0);
    test_kernel<<<gridSize, blockSize>>>(d_a, 0, OUTER_LOOPS);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("Cold Run costs  %f(ms)\n",time_elapsed);

    // Execute the kernel
    cudaEventRecord( start,0);
    test_kernel<<<gridSize, blockSize>>>(d_a, 0, OUTER_LOOPS);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("Warm Run costs  %f(ms)\n",time_elapsed);

    //This GPU in 1.65Ghz 
	int latency = int(time_elapsed * (double)1.650*1e6 / ((double)INNER_LOOPS * (double)OUTER_LOOPS));
	printf("latency for this GPU (1.65Ghz):  %d cycles \n", latency );

    // Copy result back to host
    cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost );
 
    // free device memory
    cudaFree(d_a);
 
    // free host memory
    free(h_a);
 
    return 0;
}
