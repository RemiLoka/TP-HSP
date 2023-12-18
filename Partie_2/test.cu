#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 5
#define HEIGHT 5
#define KERNEL_SIZE 3
#define C1_DEPTH 1
#define C1_WIDTH (WIDTH - KERNEL_SIZE + 1)
#define C1_HEIGHT (HEIGHT - KERNEL_SIZE + 1)

// Add the definition of your conv2D_kernel here
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


void printMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%f ", matrix[i * height + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main() {
    // Define input matrix and kernel
    float h_input[WIDTH * HEIGHT] = {0};
    h_input[5] = 0.5; // Middle element set to 1
    float h_kernel[KERNEL_SIZE * KERNEL_SIZE] = {1, 1, 1, 1, 1, 1, 1, 1, 1}; // Simple 3x3 kernel
    float h_output[C1_DEPTH * C1_WIDTH * C1_HEIGHT] = {0};

    // Allocate memory on GPU
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMalloc(&d_output, sizeof(h_output));
    
    // Copy data to GPU
    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    // Run the kernel
    dim3 blockDims(16, 16);
    dim3 gridDims((C1_WIDTH + blockDims.x - 1) / blockDims.x, (C1_HEIGHT + blockDims.y - 1) / blockDims.y);
    conv2D_kernel<<<gridDims, blockDims>>>(d_input, d_output, d_kernel, WIDTH, HEIGHT, KERNEL_SIZE, C1_DEPTH);

    // Copy result back to CPU
    cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost);

    // Print the input matrix
    printf("Input Matrix:\n");
    printMatrix(h_input, WIDTH, HEIGHT);

    // Print the kernel
    printf("Kernel Matrix:\n");
    printMatrix(h_kernel, KERNEL_SIZE, KERNEL_SIZE);

    // Print the output matrix
    printf("Output Matrix:\n");
    printMatrix(h_output, C1_WIDTH, C1_HEIGHT);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}