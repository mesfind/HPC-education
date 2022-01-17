#include <stdio.h>
#include <iostream>

/*
// created by: Isayah Reed
// Before compiling this program, set the cuda library path. Example:
// $> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
// Next, compile using nvcc:
// $> nvcc hello.c
*/

#define ARRAY_SIZE 10

__global__ void cuda_hello()
  { printf("Hello world!\n"); }

__global__ void printThread()
  {  printf("%i ",threadIdx.x); }

__global__ void increment(const int a, int *b)
  {  *b += a;  }

__global__ void incrementArray(int *array, const int b)
  {  int idx = blockIdx.x  * blockDim.x + threadIdx.x;
     array[idx] += b;  }

int main() {
   // This file needs to be named *.cu or else nvcc compiler will not
   // recognize function<<<x,x>>>(), because it is not standard C/C++ syntax
    cuda_hello<<<1,1>>>();

   // The previous function will not give an output unless/until there
   // is additional computation on the GPU.

  int a=2, *dev_a;   // 'a' will be host data, dev_a will be device data
    // create device/GPU data for dev_a
  cudaMalloc((void**)&dev_a, sizeof(int));
    // copy data from host/CPU to device/GPU
  cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyDeviceToHost);
    // load and execute GPU kernel
  increment<<<1,1>>>(5, dev_a);
    // copy data back to CPU
  cudaMemcpy(&a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << a << std::endl;    // output should show a=7
  cudaFree(dev_a);

  int array[ARRAY_SIZE] = {0};
  int *dev_array;
  cudaMalloc((void**)&dev_array, sizeof(int)*ARRAY_SIZE);
  cudaMemcpy(dev_array, array, sizeof(int)*ARRAY_SIZE, cudaMemcpyHostToDevice);
  incrementArray<<<1,ARRAY_SIZE>>>(dev_array,1);
  cudaMemcpy(array, dev_array, sizeof(int)*ARRAY_SIZE, cudaMemcpyDeviceToHost);
  for(int i=0; i<ARRAY_SIZE; i++)
    std::cout << array[i] << " ";    // all array elements should be 1
  std::cout << std::endl;
  increment<<<1,1>>>(-1, &dev_array[ARRAY_SIZE-1]);
  cudaMemcpy(array, dev_array, sizeof(int)*ARRAY_SIZE, cudaMemcpyDeviceToHost);
  for(int i=0; i<ARRAY_SIZE; i++)
    std::cout << array[i] << " ";    // last element should be 0
  std::cout << std::endl;


   // Cuda threads are dentified with threadIdx.x, threadIdx.y, threadIdx.z.
   // This function is 1D, so only uses threadIdx.x
  printThread<<<1,5>>>();

   // The printThread kernel does not require waiting for computation, so we
   //  must wait until it is complete to avoid sync issues
  cudaDeviceSynchronize();
  std::cout << std::endl;

  printThread<<<3,5>>>();
  cudaDeviceSynchronize();
  std::cout << std::endl;

  cudaFree(dev_array);

  return 0;
}
