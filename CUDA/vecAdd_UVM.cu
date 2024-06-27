// This example demonstrates parallel floating point vector
// addition with a simple __global__ function.

#include <stdlib.h>
#include <stdio.h>


// this kernel computes the vector sum c = a + b
// each thread performs one pair-wise addition
__global__ void vector_add(const float *a,
                           const float *b,
                           float *c,
                           const size_t n)
{
  // compute the global element index this thread should process
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

  // avoid accessing out of bounds elements
  if(index < n)
  {
    // sum elements
    c[index] = a[index] + b[index];
  }
}


int main(void)
{
  // create arrays of 1M elements
  const int num_elements = 1<<20;

  // compute the size of the arrays in bytes
  const int num_bytes = num_elements * sizeof(float);

  // points to used arrays
  float *array_a = NULL;
  float *array_b = NULL;
  float *array_c = NULL;

  // cudaMalloc the UVM shared arrays
  cudaMallocManaged(&array_a, num_bytes);
  cudaMallocManaged(&array_b, num_bytes); 
  cudaMallocManaged(&array_c, num_bytes);

  // if any memory allocation failed, report an error message
  if(array_a == 0 || array_b == 0 || array_c == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // initialize host_array_a & host_array_b
  for(int i = 0; i < num_elements; ++i)
  {
    // make array a a linear ramp
    array_a[i] = (float)i;

    // make array b random
    array_b[i] = (float)rand() / RAND_MAX;
  }

  // compute c = a + b on the device
  const size_t nThreads = 256;
  size_t nBlocks = num_elements / nThreads;

  // deal with a possible partial final block
  if(num_elements % nThreads) ++nBlocks;

  // launch the kernel
  vector_add<<<nBlocks, nThreads>>>(array_a, array_b, array_c, num_elements);
  cudaDeviceSynchronize();

  // print out the first 10 results
  for(int i = 0; i < 10; ++i)
  {
    printf("result %d: %1.1f + %7.1f = %7.1f\n", i, array_a[i], array_b[i], array_c[i]);
  }

  // deallocate memory
  cudaFree(array_a);
  cudaFree(array_b);
  cudaFree(array_c);

  return 0;
}


/*
UVM - unified virtual memory. Creates the illusion of having a shared memory by using a similar approach used in virtual memory.
Not a big speedup due to page faults unless pages are pre-fetched.

TODO - removed host + device allocated arrays and removed mem copies as the memory is "shared"

36  float *array_a = NULL;
37  float *array_b = NULL;
38  float *array_c = NULL;

46  if(array_a == 0 || array_b == 0 || array_c == 0)

56  array_a[i] = (float)i;

59  array_b[i] = (float)rand() / RAND_MAX;

70  vector_add<<<nBlocks, nThreads>>>(array_a, array_b, array_c, num_elements);

75 printf("result %d: %1.1f + %7.1f = %7.1f\n", i, array_a[i], array_b[i], array_c[i]);

79  cudaFree(array_a);
80  cudaFree(array_b);
81  cudaFree(array_c);

[rm.barbosa@deeplearning01 6_vectorAdd_UVM]$ ./vecAdd_UVM
result 0: 0.0 +     0.8 =     0.0
result 1: 1.0 +     0.4 =     0.0
result 2: 2.0 +     0.8 =     0.0
result 3: 3.0 +     0.8 =     0.0
result 4: 4.0 +     0.9 =     0.0
result 5: 5.0 +     0.2 =     0.0
result 6: 6.0 +     0.3 =     0.0
result 7: 7.0 +     0.8 =     0.0
result 8: 8.0 +     0.3 =     0.0
result 9: 9.0 +     0.6 =     0.0


with 
71  cudaDeviceSynchronize();

[rm.barbosa@deeplearning01 6_vectorAdd_UVM]$ ./vecAdd_UVM
result 0: 0.0 +     0.8 =     0.8
result 1: 1.0 +     0.4 =     1.4
result 2: 2.0 +     0.8 =     2.8
result 3: 3.0 +     0.8 =     3.8
result 4: 4.0 +     0.9 =     4.9
result 5: 5.0 +     0.2 =     5.2
result 6: 6.0 +     0.3 =     6.3
result 7: 7.0 +     0.8 =     7.8
result 8: 8.0 +     0.3 =     8.3
result 9: 9.0 +     0.6 =     9.6
*/
