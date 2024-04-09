#include <iostream>

__global__ void helloCudaKernel() {
    printf("Hello, CUDA!\n");
}

void runHelloCuda() {
    helloCudaKernel<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to complete
}
