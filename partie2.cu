#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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


void initMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void zeroMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = 0.0f;
    }
}

void conv2D(float *input, float *output, float *kernel, int width, int height, int kernelSize) {
    int depth = C1_DEPTH;
    for (int d = 0; d < depth; d++) {
        for (int x = 0; x < width - kernelSize + 1; x++) {
            for (int y = 0; y < height - kernelSize + 1; y++) {
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
}

void subSample(float *input, float *output, int width, int height) {
    int depth = C1_DEPTH;
    int newWidth = width / 2;
    int newHeight = height / 2;
    for (int d = 0; d < depth; d++) {
        for (int x = 0; x < newWidth; x++) {
            for (int y = 0; y < newHeight; y++) {
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
}

void MatrixPrint(float* matrix, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%f ", matrix[i * height + j]);
        }
        printf("\n");
    }
}

__device__ float activation_tanh(float M) {
    return tanh(M);
}


int main() {
    srand(time(NULL));

    initMatrix(raw_data, WIDTH * HEIGHT);
    zeroMatrix(C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT);
    zeroMatrix(S1_data, C1_DEPTH * S1_WIDTH * S1_HEIGHT);
    initMatrix(C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE);

    conv2D(raw_data, C1_data, C1_kernel, WIDTH, HEIGHT, KERNEL_SIZE);
    //activation_tanh(C1_data);

    subSample(C1_data, S1_data, C1_WIDTH, C1_HEIGHT);

    MatrixPrint(C1_data, C1_WIDTH, C1_HEIGHT);

    return 0;
}
