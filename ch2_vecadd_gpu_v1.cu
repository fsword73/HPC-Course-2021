#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
 
// CUDA kernel. One thread execute 1 workload of C = A + B
__global__ void vecAdd(float *a, float *b, float *c, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x; 
    
    if (i < n)
        c[i] = a[i] * a[i] + b[i] * b[i];
}

#define ARR_SIZE (1024*16) 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = ARR_SIZE;
 
    // Host vectors
    float *h_a;
    float *h_b;
    float *h_c;
 
    // Device input vectors
    float *d_a;
    float *d_b;
    float *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(float);
 
    // Allocate memory  on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
 
    // Allocate memory  on GPU

    gpuErrchk(cudaMalloc(&d_a, bytes));
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    int i;
    // Initialize on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sinf(i);
        h_b[i] = cosf(i);
    }
 
    // Copy from host to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // total blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
    // Copy result back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Result must be close to 1
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/(double)n);
 
    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // free host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}

 
