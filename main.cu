#include <stdio.h>

// __global__ : declaration specifier (aka declspec): a C-language construct => CUDA knows this is a kernel and not CPU code
__global__ void rgba_to_greyscale(const uchar4* const rbgaImage, unsigned char* const greyImage, int numRows, int numCols) {
    // TO-DO:
    // Fill in the kernel to convert from colour to greyscale
    // the mapping from components of a uchar4 to RGBA is:
    // .x -> R; .y -> G; .z -> B; .w -> A
    //
    // The output (greyImage) at each pixel should be the result of
    // applying the formula: output = .299f * R + .587f *G + .114f * B;
    // Note: We will be ignoring the alpha channel for this conversion

    // First create a mapping from the 2D block and grid locations
    // to an absolute 2D location in the image, then use that to
    // calculate a 1D offset
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (y < numCols && x < numRows) {
        int index = numRows * y + x;
        uchar4 colour = rbgaImage[index];
        unsigned char grey = (unsigned char)(0.299f * colour.x + 0.587f * colour.y + 0.114f * colour.z);
        greyImage[index] = grey;
    }
}

void your_rgba_to_greyscale(const uchar4* const h_rgbaImage, uchar4* const d_rgbaImage, 
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
    // You must fill in the correct sizes for the blockSize and gridSize
    // currently only one block with one thread is being launched
    // const dim3 blockSize(1, 1, 1); // TO-DO
    int blockWidth = 32;
    const dim3 blockSize(blockWidth, blockWidth, 1);

    // const dim3 gridSize(numCols, numRows, 1); // TO-DO
    int blocksX = numRows / blockWidth + 1;
    int blocksY = numCols / blockWidth + 1;
    const dim3 gridSize(blocksX, blocksY, 1);

    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

int main(int argc, char **argv) {
    const int ARRAY_SIZE = 96;
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

    // launch the kernel on one block of 96 threads | KERNEL <<<grid of blocks, blocks of threads>>>(...) | 1, 2 or 3D - dim3(x,y,z) dim3(w,1,1) == dim3(w) == w
    // kernel<<<dim3(bx,by,bz), dim3(tx,ty,tz), shmem>>>(...) | shmem = shared mem per block in bytes -> defaults to 0
    cube<<<1, ARRAY_SIZE>>>(d_out, d_in); // tells the CPU to launch on the GPU 96 copies of the kernel on 96 threads

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
