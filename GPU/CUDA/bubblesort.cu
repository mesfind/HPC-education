#include <stdio.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <time.h>

/*
 created by: Isayah Reed
 This program gives an example of a bubble sort algorithm on GPU. It is intended
  to demonstrate GPU threads and how they differ from CPU. Note that the sort 
  algorithm is different between GPU and CPU, due to how GPU threads operate. 
  Using the CPU algorithm on GPU would be ineffiecient, both in coding 
  and performance.

 Before compiling this program, set the cuda library path. Example:
 $> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

 Next, compile using nvcc:
 $> nvcc bubblesort.cu -std=c++11
*/

#define SIZE 20 // Number of variables to be sorted
#define SEED 99 // random number seed, for reproduceability

using namespace std;

__global__ void sortGPU(int *array)
{
  __shared__ bool sorted;
  int idx = blockIdx.x  * blockDim.x + threadIdx.x;
  int temp=-1;
  do
  {
    sorted = true;
    __syncthreads();
    // swap odd indices
    if((idx%2==1) && (idx!=SIZE-1))
      if(array[idx] > array[idx+1])
      { temp = array[idx];
        array[idx] = array[idx+1];
        array[idx+1] = temp;
        sorted = false;
      }
    __syncthreads();

    // swap even indices
    if((idx%2==0) && (idx!=SIZE-1))
      if(array[idx] > array[idx+1])
      { temp = array[idx];
        array[idx] = array[idx+1];
        array[idx+1] = temp;
        sorted = false;
      }
    __syncthreads();
  }
  while(sorted==false);
}


bool sortCPU(int *array)
{
  bool sorted = true;
  int *first=&array[0], *second=&array[1], temp=-1;
  for(int i=0; i<SIZE-1; i++)
  {
    if(*first > *second)
    { temp = *first;
      *first = *second;
      *second = temp;
      sorted = false;
    }
    first++;
    second++;
  }
  return sorted;
}

int main() {

  int *dev_array, *array = new int[SIZE]{0};

  cudaMalloc((void**)&dev_array, SIZE*sizeof(int));

  srand(SEED);
  // Use this line if you dont care about reproducing results
  // srand(time(NULL));

  //Initialize array
  for(int i=0; i<SIZE; i++)
    array[i] = rand()%1000;

  cout<<"Initial array, GPU: "<<endl<<array[0];
  for(int i=1; i<SIZE; i++)
    cout<<", "<<array[i];
  cout<<endl;

  cudaMemcpy(dev_array, array, SIZE*sizeof(int), cudaMemcpyHostToDevice);
  sortGPU<<<1,SIZE>>>(dev_array);
  cudaMemcpy(array, dev_array, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  cout<<"Final pass, GPU: "<<endl<<array[0];
  for(int i=1; i<SIZE; i++)
    cout<<", "<<array[i];
  cout<<"\n"<<endl;

  //Initialize array
  for(int i=0; i<SIZE; i++)
    array[i] = rand()%1000;

  cout<<"Initial array, CPU: "<<endl<<array[0];
  for(int i=1; i<SIZE; i++)
    cout<<", "<<array[i];
  cout<<"\n"<<endl;

  while(!sortCPU(array))
    ;

  cout<<"Final pass, CPU: "<<endl<<array[0];
  for(int i=1; i<SIZE; i++)
    cout<<", "<<array[i];
  cout<<endl;

  cudaFree(dev_array);
  delete [] array;
  return 0;
}
