#include <stdio.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <time.h>
#include <math.h>

/*
 created by: Isayah Reed
 This program does 90 degree matrix rotations on GPU to demonstrate
  cuda blocks. Each matrix is stored on separate cuda blocks as a 2D arrayi.

 Before compiling this program, set the cuda library path. Example:
 $> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

 Next, compile using nvcc:
 $> nvcc matrixRotation2D.cu -std=c++11
*/

#define NUM_MATRIX 3 // Number of matrices
#define DIM 5 // Matrix dimensions
#define SEED 75 // random number seed, for reproduceability

using namespace std;

void printMatrixCPU(int ***matrix)
{
  for(int x=0; x<NUM_MATRIX; x++)
  { cout<<"   Matrix "<<x+1<<":"<<endl;
    for(int y=0; y<DIM; y++)
    {
      cout<<"\t";
      for(int z=0; z<DIM; z++)
        cout<<matrix[x][y][z]<<' ';
      cout<<endl;
    }
    cout<<endl;
  }
}

__global__ void printMatrixGPU(int *matrix, const int count)
{
  int idx = (blockIdx.x*blockDim.x*blockDim.x) + (blockDim.x * threadIdx.y + threadIdx.x);
  int x,y;

  for(x=0; x<gridDim.x; x++)
  {
    if(idx + blockIdx.x == 0)   printf("   Matrix %i:\n", count);
    __syncthreads();
    if(blockIdx.x == x)//x)
    {
      for(y=0; y<blockDim.x; y++)
      {
        if((blockIdx.x + threadIdx.x + threadIdx.y) == 0)   printf("\t");
        if(threadIdx.y == y)
          printf("%i ", matrix[idx]);
        if((blockIdx.x + threadIdx.x + threadIdx.y) == 0)   printf("\n");
      }
    }
  }
}


void rotateCPU(int ***matrix)
{

  int temp=-1;
  // rotate along diagonal
  for(int x=0; x<NUM_MATRIX; x++)
    for(int y=0; y<DIM; y++)
      for(int z=0; z<y; z++)
      {
        temp = matrix[x][y][z];
        matrix[x][y][z] = matrix[x][z][y];
        matrix[x][z][y] = temp;
      }

  // rotate along middle column
  for(int x=0; x<NUM_MATRIX; x++)
    for(int y=0; y<DIM; y++)
      for(int z=0; z<DIM/2; z++)
      {
        temp = matrix[x][y][z];
        matrix[x][y][z] = matrix[x][y][DIM-z-1];
        matrix[x][y][DIM-z-1] = temp;
      }
}


__global__ void rotateGPU(int *matrix)
{
  int idx = blockDim.x * threadIdx.y + threadIdx.x;
  int offset = blockIdx.x * blockDim.x * blockDim.x;
  idx += offset;
  int tmp = -1;

  // trasnpose
  if(idx % blockDim.x <= (int)ceil(((double)blockDim.x / 2.0)))
  {
    tmp = matrix[idx];
    matrix[idx] = matrix[offset+(blockDim.x*threadIdx.x+threadIdx.y)];
    matrix[offset+(blockDim.x*threadIdx.x+threadIdx.y)] = tmp;
  }
  __syncthreads();
  // reverse rows
  if(idx % blockDim.x <= (int)ceil(((double)blockDim.x / 2.0)))
  {
    tmp = matrix[idx];
    matrix[idx] = matrix[(offset+(blockDim.x*threadIdx.y+blockDim.x)-(threadIdx.x+1))];
    matrix[(offset+(blockDim.x*threadIdx.y+blockDim.x)-(threadIdx.x+1))] = tmp;
  }
}







int main() {

  int ***matrix = new int**[NUM_MATRIX];
  int *dev_matrix;

  // create NUM_MATRIX cuda blocks, each containing a 2D DIMDIM matrix
  dim3 numBlocks(NUM_MATRIX);
  dim3 threadsPerBlock(DIM,DIM);

  srand(SEED);

  //Allocate matrices
  for(int x=0; x<NUM_MATRIX; x++)
  {
    matrix[x] = new int*[DIM];
    for(int y=0; y<DIM; y++)
    {
      matrix[x][y] = new int[DIM];
      for(int z=0; z<DIM; z++)
        {
          matrix[x][y][z] = 0;
        }
    }
  }

  //Initialize matrices
  for(int x=0; x<NUM_MATRIX; x++)
    for(int y=0; y<DIM; y++)
      for(int z=0; z<DIM; z++)
          matrix[x][y][z] = rand()%100;

  cudaMalloc((void**)&dev_matrix, NUM_MATRIX*DIM*DIM*sizeof(int));

  // Dynamically allocating ***matrix does not necessarily create contiguous
  //  memory, so we must copy ***matrix to the device by pieces
  for(int x=0; x<NUM_MATRIX; x++)
    for(int y=0; y<DIM; y++)
        cudaMemcpy(&dev_matrix[(x*DIM*DIM)+(DIM*y)], matrix[x][y], DIM*sizeof(int), cudaMemcpyHostToDevice);

  cout<<"Initial matrices, CPU: "<<endl;
  printMatrixCPU(matrix);
  cout<<"Rotated matrices, CPU: "<<endl;
  rotateCPU(matrix);
  printMatrixCPU(matrix);
  cout<<"2nd rotation, CPU: "<<endl;
  rotateCPU(matrix);
  printMatrixCPU(matrix);


  cout<<"\nInitial matrices, GPU: "<<endl;
  // While we could just copy the matrix back to host then use printMatrixCPU,
  //  the GPU version helps with visualizing the matrix layout across cuda blocks
  for(int i=0; i<NUM_MATRIX; i++)
    // Unlike cuda threads, the order of cuda block execution is not guaranteed.
    // Therefore, we launch a separate kernel for each block
    printMatrixGPU<<<1,threadsPerBlock>>>(&dev_matrix[i*DIM*DIM],i+1);
  // To demonstrate why we need to launch separate kernels for each cuda
  //  block, uncomment the next line and run with NUM_MATRIX>1
  //printMatrixGPU<<<numBlocks,threadsPerBlock>>>(dev_matrix);
  cudaDeviceSynchronize();
  
  cout<<"\nRotated matrices, GPU: "<<endl;
  rotateGPU<<<numBlocks,threadsPerBlock>>>(dev_matrix);
  cudaDeviceSynchronize();
  // Let's do the alternate way of copy-back then printMatrixCPU
  for(int x=0; x<NUM_MATRIX; x++)
    for(int y=0; y<DIM; y++)
        cudaMemcpy(matrix[x][y], &dev_matrix[(x*DIM*DIM)+(DIM*y)], DIM*sizeof(int), cudaMemcpyDeviceToHost);
  printMatrixCPU(matrix);
  
  cout<<"\n2nd rotation, GPU: "<<endl;
  rotateGPU<<<numBlocks,threadsPerBlock>>>(dev_matrix);
  cudaDeviceSynchronize();
  for(int i=0; i<NUM_MATRIX; i++)
    printMatrixGPU<<<1,threadsPerBlock>>>(&dev_matrix[i*DIM*DIM],i+1);
  cudaDeviceSynchronize();
  cout<<"\n"<<endl;


  cudaFree(dev_matrix);
  delete [] matrix;
  return 0;
}
                                
