#include <stdio.h>

// Using different memory spaces in CUDA

/********************
 * using local mem  *
 ********************/

// A __device__ or __global__ function (kernel) runs on the GPU
__global__ void use_local_memory_GPU(float in)
{
    float f;    // Var "f" is in local mem & private to each thread
    f = in;     // Param "in" is in local mem & private to each thread
    // ... real code would presumably do other stuff here ...
}

/********************
 * using global mem *
 ********************/

// A __global__ function (kernel) runs on the GPU & can be called from host
__global__ void use_global_memory_GPU(float *array)
{
    // "array" is a pointer into global memory on the device
    array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}

/********************
 * using shared mem *
 ********************/

 // Omitting out-of-bounds checks
 __global__ void use_shared_memory_GPU(float *array)
{
    // Local variables, private to each thread
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[128];

    // Copy data from "array" in global memory to sh_arr in shared memory
    // Here, each thread is responsible for copying a single element
    sh_arr[index] = array[index];

    __syncthreads(); // Ensure all the writes to shared memory have completed

    // Now, sh_arr is fully populated
    for(i = 0; i < index; i++)
    {
        sum += sh_arr[i];
    }
    average = sum / (index + 1.0f);

    // If array[index] is greater than the average of array[0..index-1], replace with average
    // Since array[] is in global memory, this change will be seen by the host (and potentially other thread blocks, if any)
    if (array[index] > average)
    {
        array[index] = average;
    }
}


int main(int argc, char **argv)
{
    /*
     * First, call a kernel that shows using local mem
     */
    use_local_memory_GPU<<1, 128>>(2.0f);

    /*
     * Next, call a kernel that shows using global mem
     */
    float h_arr[128]; // Convention: h_ vars live on host
    float* d_arr; // Convention: d_ vars live on device (GPU global mem)

    // Allocate global memory (128 floats) on the device, place result in "d_arr" (pointer to array)
    cudaMalloc((void **) &d_arr, sizeof(float) * 128); // Passing a pointer to var d_arr (which is itself a pointer)

    // Now copy data from host memory "h_arr" to device memory "d_arr"
    cudaMemcpy((void *) d_arr, (void *) h_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);

    // Launch the kernel (1 block of 128 threads)
    use_global_memory_GPU<<<1, 128>>>(d_arr); // Modifies the contents of array at d_arr

    // Copy the modified array back to the host, overwriting contents of h_arr
    cudaMemcpy((void *) h_arr, (void *) d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);
    
    // ... Do other stuff ...
    
    /*
     * Next, call a kernel that shows using global mem
     */

    // As before, pass in a pointer to a data in global memory
    use_shared_memory_GPU<<<1, 128>>>(d_arr);
    // Copy the modified array back to host
    cudaMemcpy((void *) h_arr, (void *) d_arr, sizeof (float) * 128, cudaMemcpyHostToDevice);

    // ... Do other stuff ...

    return 0;
}