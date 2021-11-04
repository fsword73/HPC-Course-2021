/*
please note that the series of optmiztion technology is not in official document.

All the tests are based on AMD MI25 radeon instict and AMD ROCm.
*/




#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>




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

#define THREADS_PER_BLOCK_X  64
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define OUTER_LOOPS        10000
#define INNER_LOOPS        100


__global__ void
test_kernel(int* __restrict__ buf, int stride, int mask, int outerLloops)
{

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int shared_data[16384];

	for (int i = 0; i < NUM; i += 64){
		shared_data[threadIdx.x + i] = buf[threadIdx.x + i];
	}

	int temp = (threadIdx.x * stride) & mask;
	for(int i = 0; i < outerLloops; i++)
	{
		for(int j = 0; j < INNER_LOOPS; j++)
		{
            //shared_data[temp] always equal to = 0 but compiler never knows            
			temp = ((shared_data[temp] + threadIdx.x)*stride ) & mask;
            
		}
	}
	if (temp > 0)
	{
		buf[x] = temp;
	}
}


using namespace std;

int main() {

	int* hostA;

	int* deviceA;

    cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float eventMs = 1.0f;

	
	hostA = (int*)malloc(NUM * sizeof(int));

	int* p;

	p = hostA;
	for (int i = 0; i < NUM; i += 1)
	{
		*p = 0;
		p++;
	}

	gpuErrchk(cudaMalloc((void**)& deviceA, NUM * sizeof(int)));
	gpuErrchk(cudaMemcpy(deviceA, hostA, NUM * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 64;
 
    // total blocks in grid
    gridSize = (int)ceil((int)NUM/blockSize);
    test_kernel<<<gridSize, blockSize>>>(deviceA, 1, NUM-1, OUTER_LOOPS);


	
    //for (int i = 1; i < 65; i= i*2)    
    for (int i = 4; i < 65; i+=4)
    //for (int i = 1; i < 65; i++)
    {
		cudaEventRecord(start, NULL);
        test_kernel<<<gridSize, blockSize>>>(deviceA, i, NUM-1, OUTER_LOOPS);

		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&eventMs, start, stop);

		printf("elapsed time:%f\n", eventMs);
		int latency = int(eventMs * (double)1.65 * 1e6 / ((double)INNER_LOOPS * (double)OUTER_LOOPS));
		printf("strdie = [%d], latency for this GPU(1.65Ghz):  %d cycles \n", i, latency);
	}
	gpuErrchk(cudaFree(deviceA));

	free(hostA);

	return 0;
}
