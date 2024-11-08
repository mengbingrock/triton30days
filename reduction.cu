#include <stdio.h>
#include <iostream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <random>

//#include <helper_cuda.h>


__global__ void sum_reduction_1(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


    sdata[tid] = (i < n)? g_idata[i]: 0.0f; // copy data from global memory to shared

    __syncthreads(); // synchronize all threads in the block

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void sum_reduction_2(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


    sdata[tid] = (i < n)? g_idata[i]: 0.0f; // copy data from global memory to shared

    __syncthreads(); // synchronize all threads in the block

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = tid * 2 * s;
        if (index < blockDim.x) {
            sdata[index] = sdata[index] + sdata[index + s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void sum_reduction_3(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


    sdata[tid] = (i < n)? g_idata[i]: 0.0f; // copy data from global memory to shared

    __syncthreads(); // synchronize all threads in the block

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {

        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void sum_reduction_4(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;


    sdata[tid] = (i < n)? g_idata[i] + g_idata[i + blockDim.x]: 0.0f; // copy data from global memory to shared

    __syncthreads(); // synchronize all threads in the block

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {

        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void sum_reduction_5(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sdata[tid] = 0.0;

    while (i  < n) {
      sdata[tid] += g_idata[i] + g_idata[i + blockDim.x];
      i = i + gridDim.x * blockDim.x;
    }

    __syncthreads();

    for(int stride = blockDim.x / 2; stride; stride >>= 1) {
      
      if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
      }
      __syncthreads();
    }

    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 4*1024*1024; // Number of elements (1 million)
    size_t bytes = n * sizeof(float);
    // Create a random number generator
    std::random_device rd;   // Non-deterministic seed generator
    std::mt19937 gen(rd());  // Mersenne Twister random number engine

    // Define a range for the random floats (e.g., between 0.0 and 1.0)
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Allocate memory on the host (CPU)
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // Initialize input array with some values
    for (int i = 0; i < n; ++i) {
        h_input[i] = dis(gen); // You can initialize with any value, e.g., 1.0f for easy verification
    }

    // Allocate memory on the device (GPU)
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 128; // Number of threads per block
    int gridSize = (n + blockSize  - 1) / (blockSize ); // Number of blocks

    int gridSize_2 = (n + blockSize * 2 - 1) / (blockSize * 2); // Number of blocks

    // Launch the reduction kernel
    // sum_reduction_5<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);

    // sum_reduction_4<<<gridSize_2, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);

    sum_reduction_5<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);


    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform the final reduction on the CPU (if needed)
    float final_sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        final_sum += h_output[i];
    }

    // Print the result
    std::cout << "Sum from cuda: " << final_sum << std::endl;

    float cpu_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_sum += h_input[i];
    }
    std::cout << "Sum from cpu:  " << cpu_sum << std::endl;

    // Clean up memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
