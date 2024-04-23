#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#define BLOCK_SIZE 256  // Adjust based on your GPU's capabilities

__global__ void generateKroneckerEdges(int *start, int *end, int scale, int edgefactor, unsigned long seed) {
    int num_vertices = 1 << scale;
    int num_edges = num_vertices * edgefactor;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double A = 0.57, B = 0.19, C = 0.19;
    double ab = A + B;
    double c_norm = C / (1 - (A + B));
    double a_norm = A / (A + B);

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    if (idx < num_edges) {
        int local_start = 0, local_end = 0;
        for (int ib = 0; ib < scale; ++ib) {
            bool ii_bit = curand_uniform(&state) > ab;
            bool jj_bit = curand_uniform(&state) > (c_norm * ii_bit + a_norm * !ii_bit);
            local_start += (ii_bit ? 1 : 0) * (1 << ib);
            local_end += (jj_bit ? 1 : 0) * (1 << ib);
        }
        start[idx] = local_start;
        end[idx] = local_end;
    }
}

void generateKroneckerEdgeList(int scale, int edgefactor, unsigned long seed, int* start, int* end) {
    int num_vertices = 1 << scale;
    int num_edges = num_vertices * edgefactor;
    int *d_start, *d_end;

    cudaMalloc(&d_start, num_edges * sizeof(int));
    cudaMalloc(&d_end, num_edges * sizeof(int));

    int blocks = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generateKroneckerEdges<<<blocks, BLOCK_SIZE>>>(d_start, d_end, scale, edgefactor, seed);

    cudaMemcpy(start, d_start, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(end, d_end, num_edges * sizeof(int), cudaMemcpyDeviceToHost);

    // Optionally shuffle on CPU
    // std::shuffle & std::iota for permutation if needed, as discussed

    cudaFree(d_start);
    cudaFree(d_end);
    cudaDeviceSynchronize();
}


// Allocates cuda devices by rank
int cudaInit(int rank) {
  int device_count, cE;

  // get device count and check for errors
  if ((cE = cudaGetDeviceCount(&device_count)) != cudaSuccess) {
    std::cout << " Unable to determine cuda device count, error is "<< cE << ", count is " << device_count << "\n";
    exit(-1);
  }

  if ((cE = cudaSetDevice(rank % device_count)) != cudaSuccess) {
    std::cout << " Unable to have rank " << rank << " set to cuda device " << rank % device_count << ", error is " << cE << " \n";
    exit(-1);
  }

  int device;
  cudaGetDevice(&device);

  // debugging
  // printf("rank: %d \t Attatched to CUDA Device %d\n", rank, device);

  return device_count;
}

