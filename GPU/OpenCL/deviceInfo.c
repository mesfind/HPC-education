#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

int main()
{
   cl_int err;
   int i;
   cl_uint num_platforms;
   clGetPlatformIDs(0, NULL, &num_platforms);
   if (num_platforms == 0)
   {
      printf("\nNo platforms available!\n");
      return 0;
   }

   cl_platform_id platform[num_platforms];
   clGetPlatformIDs(num_platforms, platform, NULL);
   printf("\nOpenCL platforms:\n\n");

   for (i=0; i<num_platforms; i++)
   {
      cl_char string[10240] = {0};
      clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
      printf("%s\n",string);

      clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
      printf("Vendor: %s\n", string);

      clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
      printf("Version: %s\n\n", string);
    }
    return 0;
}
