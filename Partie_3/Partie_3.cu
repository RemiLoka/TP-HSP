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
#define C3_DEPTH 16
#define C3_WIDTH 10
#define C3_HEIGHT 10
#define S3_WIDTH 5
#define S3_HEIGHT 5
#define FLATTEN_SIZE (C3_DEPTH * S3_WIDTH * S3_HEIGHT)
#define C5_SIZE 120
#define F6_SIZE 84
#define OUTPUT_SIZE 10



float raw_data[WIDTH * HEIGHT];
float C1_data[C1_DEPTH * C1_WIDTH * C1_HEIGHT];
float S1_data[C1_DEPTH * S1_WIDTH * S1_HEIGHT];
float C1_kernel[C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE];


float C3_data[C3_DEPTH * C3_WIDTH * C3_HEIGHT];
float S3_data[C3_DEPTH * S3_WIDTH * S3_HEIGHT];
float C3_kernel[C3_DEPTH * C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE];
float flatten_data[FLATTEN_SIZE];
float C5_weights[FLATTEN_SIZE * C5_SIZE];
float C5_data[C5_SIZE];
float F6_weights[C5_SIZE * F6_SIZE];
float F6_data[F6_SIZE];
float output_weights[F6_SIZE * OUTPUT_SIZE];
float output_data[OUTPUT_SIZE];




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


void fullyConnectedLayer(float *input, float *output, float *weights, int inputSize, int outputSize) {
    for (int i = 0; i < outputSize; i++) {
        output[i] = 0.0;
        for (int j = 0; j < inputSize; j++) {
            output[i] += input[j] * weights[i * inputSize + j];
        }
    }
}

void softmaxActivation(float *input, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
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

void apply_activation_tanh(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = tanh(matrix[i]);
    }
}

int main() {
    srand(time(NULL));

    initMatrix(raw_data, WIDTH * HEIGHT);
    zeroMatrix(C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT);
    zeroMatrix(S1_data, C1_DEPTH * S1_WIDTH * S1_HEIGHT);
    initMatrix(C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE);

     conv2D(raw_data, C1_data, C1_kernel, WIDTH, HEIGHT, KERNEL_SIZE);

    apply_activation_tanh(C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT);

    
    subSample(C1_data, S1_data, C1_WIDTH, C1_HEIGHT);

    printf("C1_data après convolution et activation:\n");
    MatrixPrint(C1_data, C1_WIDTH, C1_HEIGHT);
    printf("\nS1_data après sous-échantillonnage:\n");
    MatrixPrint(S1_data, S1_WIDTH, S1_HEIGHT);


    conv2D(S1_data, C3_data, C3_kernel, S1_WIDTH, S1_HEIGHT, KERNEL_SIZE);
    apply_activation_tanh(C3_data, C3_DEPTH * C3_WIDTH * C3_HEIGHT);
    subSample(C3_data, S3_data, C3_WIDTH, C3_HEIGHT);

    for (int i = 0; i < FLATTEN_SIZE; i++) {
        flatten_data[i] = S3_data[i];
    }

    initMatrix(C5_weights, FLATTEN_SIZE * C5_SIZE);
    fullyConnectedLayer(flatten_data, C5_data, C5_weights, FLATTEN_SIZE, C5_SIZE);
    apply_activation_tanh(C5_data, C5_SIZE);

    initMatrix(F6_weights, C5_SIZE * F6_SIZE);
    fullyConnectedLayer(C5_data, F6_data, F6_weights, C5_SIZE, F6_SIZE);
    apply_activation_tanh(F6_data, F6_SIZE);

    initMatrix(output_weights, F6_SIZE * OUTPUT_SIZE);
    fullyConnectedLayer(F6_data, output_data, output_weights, F6_SIZE, OUTPUT_SIZE);
    softmaxActivation(output_data, OUTPUT_SIZE);

    printf("\nOutput layer data:\n");
    MatrixPrint(output_data, 1, OUTPUT_SIZE);


    return 0;
}
