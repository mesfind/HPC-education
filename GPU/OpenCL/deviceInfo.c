
#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

int main(void)
{
   cl_int err;
   int i,j;
   size_t k;
   // Find the number of OpenCL platforms
   cl_uint num_platforms;
/*
 * A cl_platform is a specific OpenCL implementaton. Multiple vendors  
 *   have different OpenCL implementations, including both Nvidia 
 *   and Intel.
 */
   clGetPlatformIDs(0, NULL, &num_platforms);
   if (num_platforms == 0)
   {
      printf("No platforms available!\n");
      return 0;
   }
    // Create a list of platform IDs
   cl_platform_id platform[num_platforms];
   clGetPlatformIDs(num_platforms, platform, NULL);

   printf("OpenCL platforms:\n");
   printf("\n---------------\n");

   for (i=0; i<num_platforms; i++)
   {
      cl_char string[10240] = {0};
      clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
      printf("Platform[%i]: %s\n",i,string);

      clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
      printf("Vendor: %s\n", string);

      clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
      printf("Version: %s\n", string);

      cl_uint num_devices;
      clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

/*
 * A cl_device is a GPU, either internal (ex: Intel Integrated Graphics)   
 *   or external (ex: Nvidia cards)
 */
      cl_device_id device[num_devices];
      clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
      printf("Number of devices: %d\n", num_devices);

      for (j = 0; j < num_devices; j++)
      {
         printf("\t-------------------------\n");

         clGetDeviceInfo(device[j], CL_DEVICE_NAME, sizeof(string), &string, NULL);
         printf("\t\tName: %s\n", string);

         clGetDeviceInfo(device[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
         printf("\t\tVersion: %s\n", string);

         // Get Max. Compute units
         cl_uint num;
         clGetDeviceInfo(device[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
         printf("\t\tParallel Compute Units: %d\n", num);

         // Get local memory size
         cl_ulong mem_size;
         clGetDeviceInfo(device[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
         printf("\t\tLocal Memory Size: %llu KB\n", mem_size/1024);

         // Get global memory size
         clGetDeviceInfo(device[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
         printf("\t\tGlobal Memory Size: %llu MB\n", mem_size/(1024*1024));

         // Get global memory cache 
         clGetDeviceInfo(device[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &mem_size, NULL);
         printf("\t\tGlobal Memory Cache: %llu MB\n", mem_size/(1024*1024));

         // Get maximum buffer alloc. size
         clGetDeviceInfo(device[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
         printf("\t\tMax Alloc Size: %llu MB\n", mem_size/(1024*1024));

         // Get work-group size information
         size_t size;
         clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
         printf("\t\tMax Work-group Size: %ld\n", size);

         // Find the maximum dimensions of the work-groups
         err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
         // Get the max. dimensions of the work-groups
         size_t dims[num];
         err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
         printf("\t\tMax Work-group Dims: ( ");
         for (k = 0; k < num; k++)
         {
            printf("%ld ", dims[k]);
         }
         printf(")\n");

         printf("\t-------------------------\n");
      }

        printf("\n-------------------------\n");
    }

    return EXIT_SUCCESS;
}
