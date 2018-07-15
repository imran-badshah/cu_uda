#include <stdio.h>

// __global__ : declaration specifier (aka declspec): a C-language construct => CUDA knows this is a kernel and not CPU code
__global__ void square(float *d_out, float *d_in) {
    int idx = threadIdx.x; //(dim3: a C-Struct with .x .y and .z)tells each thread its index within a block
    float f = d_in[idx];
    d_out[idx] = f * f;
}

int main(int argc, char **argv) {
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    for (int i=0; i<ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    // declare GPU memory pointers
    float *d_in;
    float *d_out;

    // allocate GPU memory for the above pointers
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel on one block of 64 elements
    square<<<1, ARRAY_SIZE>>>(d_out, d_in); // tells the CPU to launch on the GPU 64 copies of the kernel on 64 threads

    // copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    for (int i=0; i<ARRAY_SIZE; i++) {
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
