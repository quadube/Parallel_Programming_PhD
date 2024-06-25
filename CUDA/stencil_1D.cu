#include <stdio.h>

#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4096*2)

__global__ void stencil_1d_simple(int *in, int *out) 
{
  // compute this thread's global index
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x + RADIUS;

  int alpha = 1; 
  int beta = 1; 

  if(i < NUM_ELEMENTS + RADIUS )
  {
     out[i] = alpha*(in[i-3] +in[i-2] + in[i-1] + in[i+1] + in[i+2] + in[i-3]) + beta * in[i];
  }
}

__global__ void stencil_1d_improved(int *in, int *out) 
{
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    int gindex = threadIdx.x + (blockIdx.x * blockDim.x + RADIUS);
    int lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];

    //Load ghost cells (halos)
    if (threadIdx.x < RADIUS) 
    {
       temp[lindex - RADIUS] = in[gindex - RADIUS];
       temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // Make sure all threads get to this point before proceeding!
    __syncthreads();   

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += temp[lindex + offset];

    // Store the result
    out[gindex] = result;
}

int main()
{
  unsigned int i;
  int N = NUM_ELEMENTS + 2 * RADIUS; 
  int h_in[N], h_out[N];
  int *d_in, *d_out;

  // Initialize host data
  for( i = 0; i < (N); ++i )
    h_in[i] = 1; // With a value of 1 and RADIUS of 3, all output values should be 7

  // Allocate space on the device
  cudaMalloc( &d_in,  N * sizeof(int)) ;
  cudaMalloc( &d_out, N * sizeof(int)) ;

  // Copy input data to device
  cudaMemcpy( d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice) ;

  stencil_1d_simple<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);
  stencil_1d_improved<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);

  cudaMemcpy( h_out, d_out, N *  sizeof(int), cudaMemcpyDeviceToHost) ;

  // Verify every out value is 7
  for( i = RADIUS; i < NUM_ELEMENTS+RADIUS; ++i )
    if (h_out[i] != RADIUS*2+1)
    {
      printf("Element h_out[%d] == %d != 7\n", i, h_out[i]);
      break;
    }

  if (i == NUM_ELEMENTS+RADIUS)
    printf("SUCCESS!\n");

  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

/*
FIXME1 - hardcoded for radius 3
17   out[i] = alpha*(in[i-3] +in[i-2] + in[i-1] + in[i+1] + in[i+2] + in[i-3]) + beta * in[i];

[rm.barbosa@deeplearning01 4_Stencil1D]$ ./stencil_1D 
SUCCESS!

FIXME2
23 __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

FIXME3
25  int gindex = threadIdx.x + (blockIdx.x * blockDim.x + RADIUS);

FIXME4
26  int lindex = threadIdx.x + RADIUS;

FIXME5
34     temp[lindex - RADIUS] = in[gindex - RADIUS];
35     temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];

[rm.barbosa@deeplearning01 4_Stencil1D]$ ./stencil_1D 
Element h_out[3] == 2 != 7

FIXME6 - this way we make sure threads 0 and 1 have finished because they have extra work
39    __syncthreads();  

[rm.barbosa@deeplearning01 4_Stencil1D]$ ./stencil_1D 
SUCCESS!
*/