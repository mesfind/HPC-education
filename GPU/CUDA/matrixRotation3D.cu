#include <stdio.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <time.h>

/*
 created by: Isayah Reed
 This program does 90 degree matrix rotations on GPU to demonstrate
  cuda threads. Each matrix is stored as a 2D array along a single 3D cuda block

 The rotation algorithm is different for CPU vs GPU, to help with validation.

 Before compiling this program, set the cuda library path. Example:
 $> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

 Next, compile using nvcc:
 $> nvcc matrixRotation3D.cu -std=c++11
*/

#define NUM_MATRIX 2 // Number of matrices
#define DIM 4        // Matrix dimensions
#define SEED 75      // random number seed, for reproduceability

using namespace std;

__global__ void rotateGPU(int *matrix)
{
  int idx = (blockDim.y * threadIdx.y) + threadIdx.x;
  int offset = (blockDim.y*blockDim.y*threadIdx.z);
  idx += offset;
  int tmp = -1;

  // trasnpose
  if(idx % blockDim.x <= (int)ceil(((double)blockDim.x / 2.0)))
  {
    tmp = matrix[idx];
    matrix[idx] = matrix[offset + (blockDim.y*threadIdx.x) + threadIdx.y];
    matrix[offset + (blockDim.y*threadIdx.x) + threadIdx.y] = tmp;
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


void rotateCPU(int matrix[NUM_MATRIX][DIM][DIM])
{
  int temp=-1;

  // rotate along diagonal
  for(int x=0; x<NUM_MATRIX; x++)
    for(int y=0; y<DIM; y++)
      for(int z=0; z<y; z++)
      {
        temp = matrix[x][y][z];
        matrix[x][y][z] = matrix[x][z][y];
        matrix[x][y][DIM-z-1] = temp;
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

void printMatrix(int matrix[NUM_MATRIX][DIM][DIM])
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

__global__ void printMatrixGPU(int *matrix)
{
  int idx = blockDim.y * threadIdx.y + threadIdx.x;
  int offset = blockDim.y * blockDim.y * threadIdx.z;
  idx += offset;
  int x,y;

  for(x=0; x<blockDim.z; x++)
  {
    if(idx == 0)   printf("\n   Matrix %i:\n", x+1);
    __syncthreads();
    if(threadIdx.z == x)
    {
      for(y=0; y<blockDim.y; y++)
      {
        if(threadIdx.x+threadIdx.y == 0)   printf("\t");
        if(threadIdx.y == y)
          printf("%i ", matrix[idx]);
        if(threadIdx.x+threadIdx.y == 0)   printf("\n");
      }
    }
  }

}




int main() {

  int matrix[NUM_MATRIX][DIM][DIM];
  int *dev_matrix;

  // place 2D DIMxDIM matrices along NUM_MATRIX 3D cuda threads
  dim3 threadsPerBlock(DIM,DIM,NUM_MATRIX);

  srand(SEED);

  //Initialize matrices
  for(int x=0; x<NUM_MATRIX; x++)
    for(int y=0; y<DIM; y++)
      for(int z=0; z<DIM; z++)
          matrix[x][y][z] = rand()%100;

  cudaMalloc((void**)&dev_matrix, NUM_MATRIX*DIM*DIM*sizeof(int));
  cudaMemcpy(dev_matrix, &matrix[0][0][0],NUM_MATRIX*DIM*DIM*sizeof(int), cudaMemcpyHostToDevice);

  cout<<"Initial matrices, CPU: "<<endl;
  printMatrix(matrix);
  rotateCPU(matrix);
  cout<<"Rotated matrices, CPU: "<<endl;
  printMatrix(matrix);
  rotateCPU(matrix);
  cout<<"2nd rotation, CPU: "<<endl;
  printMatrix(matrix);


  cout<<"Initial matrices, GPU: "<<endl;
  printMatrixGPU<<<1,threadsPerBlock>>>(dev_matrix);
  cudaDeviceSynchronize();

  rotateGPU<<<1,threadsPerBlock>>>(dev_matrix);
  cudaDeviceSynchronize();
  cout<<"Rotated matrices, GPU: "<<endl;
  cudaMemcpy(&matrix[0][0][0],dev_matrix,NUM_MATRIX*DIM*DIM*sizeof(int), cudaMemcpyDeviceToHost);
  printMatrix(matrix);

  rotateGPU<<<1,threadsPerBlock>>>(dev_matrix);
  cudaDeviceSynchronize();
  cout<<"2nd rotation, GPU: "<<endl;
  cudaMemcpy(&matrix[0][0][0],dev_matrix,NUM_MATRIX*DIM*DIM*sizeof(int), cudaMemcpyDeviceToHost);
  printMatrix(matrix);
  cout<<endl;


  cudaFree(dev_matrix);
  return 0;
}

