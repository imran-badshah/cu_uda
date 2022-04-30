#include <stdio.h>
#include "gputimer.h"

#define NUM_THREADS 1000000
#define ARRAY_SIZE 10
#define BLOCK_WIDTH 1000

using stdio

void print_array(int *array, int size)
{
    cout << &array;
}

__global__ void increment_atomic()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % ARRAY_SIZE;
    atomicAdd(& g[i], 1);
}

int main(int argc, char **argv)
{
    // GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n", NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);
    int* arr = []
    print_array()
    
}