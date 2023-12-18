#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define WIDTH 32
#define HEIGHT 32
#define KERNEL_SIZE 5
#define C1_DEPTH 6
#define C1_WIDTH 28
#define C1_HEIGHT 28
#define S1_WIDTH 14
#define S1_HEIGHT 14

float raw_data[WIDTH * HEIGHT];
float C1_data[C1_DEPTH * C1_WIDTH * C1_HEIGHT];
float S1_data[C1_DEPTH * S1_WIDTH * S1_HEIGHT];
float C1_kernel[C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE];




__global__ void initMatrix_kernel(float *matrix, int size, bool isZero, unsigned long long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, index, 0, &state);

    if (index < size) {
        matrix[index] = isZero ? 0.0f : curand_uniform(&state);
    }
}


__global__ void conv2D_kernel(float *input, float *output, float *kernel, int width, int height, int kernelSize, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - kernelSize + 1 && y < height - kernelSize + 1) {
        for (int d = 0; d < depth; d++) {
            float sum = 0.0;
            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    sum += input[(x + i) * height + (y + j)] * kernel[d * kernelSize * kernelSize + i * kernelSize + j];
                }
            }
            output[d * (width - kernelSize + 1) * (height - kernelSize + 1) + x * (height - kernelSize + 1) + y] = sum;
        }
    }
}


__global__ void subSample(float *input, float *output, int width, int height, int newWidth, int newHeight, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < newWidth && y < newHeight) {
        for (int d = 0; d < depth; d++) {
            float sum = 0.0;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    sum += input[d * width * height + (2 * x + i) * height + 2 * y + j];
                }
            }
            output[d * newWidth * newHeight + x * newHeight + y] = sum / 4.0;
        }
    }
}


void MatrixPrint(float* matrix, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%f ", matrix[i * height + j]);
        }
        printf("\n");
    }
}

__global__ void apply_activation_tanh_kernel(float *matrix, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        matrix[index] = tanh(matrix[index]);
    }
}


int main() {

    // Allocate memory on GPU
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;
    cudaMalloc(&d_raw_data, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT * sizeof(float));
    cudaMalloc(&d_S1_data, C1_DEPTH * S1_WIDTH * S1_HEIGHT * sizeof(float));
    cudaMalloc(&d_C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // Initialize matrices on GPU
    unsigned long long seed = time(NULL); 
    dim3 blockSize(256);
    dim3 gridSize((WIDTH * HEIGHT + blockSize.x - 1) / blockSize.x);
    initMatrix_kernel<<<gridSize, blockSize>>>(d_raw_data, WIDTH * HEIGHT, false,seed);
    initMatrix_kernel<<<gridSize, blockSize>>>(d_C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE, false,seed);
    initMatrix_kernel<<<gridSize, blockSize>>>(d_C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT, true,seed);
    initMatrix_kernel<<<gridSize, blockSize>>>(d_S1_data, C1_DEPTH * S1_WIDTH * S1_HEIGHT, true,seed);

    // Perform convolution
    dim3 convBlock(16, 16);
    dim3 convGrid((C1_WIDTH - KERNEL_SIZE + 1 + convBlock.x - 1) / convBlock.x, 
                  (C1_HEIGHT - KERNEL_SIZE + 1 + convBlock.y - 1) / convBlock.y);
    conv2D_kernel<<<convGrid, convBlock>>>(d_raw_data, d_C1_data, d_C1_kernel, WIDTH, HEIGHT, KERNEL_SIZE, C1_DEPTH);

    // Apply activation
    gridSize = dim3((C1_DEPTH * C1_WIDTH * C1_HEIGHT + blockSize.x - 1) / blockSize.x);
    apply_activation_tanh_kernel<<<gridSize, blockSize>>>(d_C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT);

    // Perform subsampling
    dim3 subBlock(16, 16);
    dim3 subGrid((S1_WIDTH + subBlock.x - 1) / subBlock.x, 
                 (S1_HEIGHT + subBlock.y - 1) / subBlock.y);
    subSample<<<subGrid, subBlock>>>(d_C1_data, d_S1_data, C1_WIDTH, C1_HEIGHT, S1_WIDTH, S1_HEIGHT, C1_DEPTH);

    // Copy results back to CPU
    cudaMemcpy(C1_data, d_C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, C1_DEPTH * S1_WIDTH * S1_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);


    // Print results
    printf("Raw data:\n");
    MatrixPrint(raw_data, WIDTH, HEIGHT);
    printf("\nC1 data:\n");
    MatrixPrint(C1_data, C1_WIDTH, C1_HEIGHT);
    printf("\nS1 data:\n");
    MatrixPrint(S1_data, S1_WIDTH, S1_HEIGHT);
    
    // Free GPU memory
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C1_kernel);

    return 0;
}
