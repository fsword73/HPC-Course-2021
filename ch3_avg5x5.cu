
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

#define THREADS_PER_BLOCK_X  32
#define THREADS_PER_BLOCK_Y  32
#define THREADS_PER_BLOCK_Z  1

#define ARR_SIZE   (1024*1024)


//local block size,  (256, 1)
//total threads (H * W, 1)
#define CLAMP(a,b,c)  min(max(a,b),c)

__global__ void avg_filter5x5(int* A, int* B,  int h, int w )
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >=w || y >= h) return;

    //read pixels
    float avg_filter[]={       
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25
      };
    int data[5*5];
    if(x == 0 || y==0 || x==(w-1) || y== (h-1) )
    {
        for(int i=0; i< 5; i++){
            for(int j=0; j< 5; j++){
                int xx = CLAMP(x+j-2,0,w-1);
                int yy = CLAMP(y+i-2,0,h-1);
                data[i*5+j] = A[xx + yy * w];
            }//forj
        }//fori
    }
    else
    {
        for(int i=0; i< 5; i++){
            for(int j=0; j< 5; j++){
                int xx = x+j-1;
                int yy = y+i-1;
                data[i*5+j] = A[xx + yy * w];
            }//forj
        }//fori
    }
    float sum=0;
    for(int i=0; i< 5; i++){
        for(int j=0; j< 5; j++){
            sum +=data[j+i *5] * avg_filter[j+i*5];
        }//forj
    }//fori
    B[x + y *w ] =(int)sum;
}

__global__ void avg_filter5x5_tile4x4(int* A, int* B,  int h, int w ){
    int x = (blockDim.x * blockIdx.x + threadIdx.x)*4;
    int y = (blockDim.y * blockIdx.y + threadIdx.y)*4;
    if(x >=w || y >= h) return;

    //read pixels
    int data[8*8];
    if(x == 0 || y==0 || (x+3)==(w-1) || (y+3)== (h-1) ){
        for(int i=0; i< 8; i++){
            for(int j=0; j< 8; j++){
                int xx = CLAMP(x+j-1,0,w-1);
                int yy = CLAMP(y+i-1,0,h-1);
                data[i*8+j] = A[xx + yy * w];
            }
        }
    }
    else 
    {
          for(int i=0; i< 8; i++){
            for(int j=0; j< 8; j++){
                int xx = x+j-1;
                int yy = y+i-1;
                data[i*8+j] = A[xx + yy * w];
            }
        }      
    }
    float avg_filter[]={      
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25,
        1.0f/25,1.0f/25,1.0f/25,1.0f/25,1.0f/25
      };    
    float sum[4*4];
    for(int s=0; s<4; s++)
        for(int t=0; t<4; t++){
        int sidx = s*4 + t;
           sum[sidx] = 0;
           for(int i=0; i< 5; i++){
                for(int j=0; j< 5; j++){
                    int xx = j+t;
                    int yy = i+s;
                    sum[sidx] +=data[ xx+yy * 8] * avg_filter[j+i*5];
                }//forj
            }//fori
            int xx = t + x;
            int yy = s + y;
            B[xx + yy * w] = (int)sum[t+s*4];
        }//fort
}



void gen_chessboard(int* img, int w, int h, int boxsize){
    for(int y=0; y < h; y++)
        for(int x=0; x < w; x++){
            int x_even, y_even;
            x_even = (x/boxsize)%2;
            y_even = (y/boxsize) %2; 
            if((x_even + y_even)== 0 || (x_even + y_even==2) ){
                img[x + y * w] = 0;
            } 
            else{
                img[x + y * w] = 255;
            }
        }
}
void save_ppm(int* img, int w, int h, const char* fname){
    FILE* f= fopen(fname, "w");
    fprintf(f, "P2 %d %d 255\n", w, h);
    for(int y=0; y < h; y++)
        for(int x=0; x < w; x+=16){
            for(int l=0; l<15;l++)
                fprintf(f, "%d ", img[x +l + y *w]);
            fprintf(f, "%d\n", img[x+15 + y *w]);
    }
    fclose(f);
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
 
    // Allocate memory  on host
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
 
    // Allocate memory  on GPU
    gpuErrchk(cudaMalloc(&d_a, bytes));
    gpuErrchk(cudaMalloc(&d_b, bytes));
 
    // Initialize on host
    gen_chessboard(h_a,1024,1024,32);
    save_ppm(h_a, 1024,1024,"chess.ppm");
    
    // Copy from host to device
    gpuErrchk(cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice));

    //block/grid
    dim3  blockSize(32,32), gridSize(32,32);
    
    
    //Profile GPU time
    float time_elapsed=0;
    cudaEvent_t start,stop;

    cudaEventCreate(&start);    
    cudaEventCreate(&stop);
    
    // Execute the kernel
    cudaEventRecord( start,0);
    avg_filter5x5<<<gridSize, blockSize>>>(d_a,d_b,1024,1024);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("Run costs  %f(ms)\n",time_elapsed);

    // Execute the kernel
    cudaEventRecord( start,0);
    dim3 gridsize2(8,8);
    avg_filter5x5_tile4x4<<<gridSize, gridsize2>>>(d_a,d_b,1024,1024);
    cudaEventRecord( stop,0);
    
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time_elapsed,start,stop);    
    printf("tile4x4 Run costs  %f(ms)\n",time_elapsed);


    // Copy result back to host
    cudaMemcpy( h_b, d_b, bytes, cudaMemcpyDeviceToHost );
    save_ppm(h_b, 1024,1024,"chess_avg5x5.ppm");
    
    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
 
    // free host memory
    free(h_a);
    free(h_b);
 
    return 0;
}

