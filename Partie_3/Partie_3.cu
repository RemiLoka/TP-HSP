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
#define OUTPUT_SIZE 10
#define FLATTEN_SIZE (C3_DEPTH * S3_WIDTH * S3_HEIGHT)
#define C5_SIZE 120
#define F6_SIZE 84
#define OUTPUT_SIZE 10


#define NUM_IMAGES 60000



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
float *d_raw_data;


float images[NUM_IMAGES][WIDTH * HEIGHT];


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

__global__ void conv2D(float *input, float *output, float *kernel, int width, int height, int kernelSize, int depth) {
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

__global__ void subSample(float *input, float *output, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int newWidth = width / 2;
    int newHeight = height / 2;

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



__global__ void fullyConnectedLayer(float *input, float *output, float *weights, int inputSize, int outputSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < outputSize) {
        float sum = 0.0;
        for (int j = 0; j < inputSize; j++) {
            sum += input[j] * weights[index * inputSize + j];
        }
        output[index] = sum;
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

__global__ void apply_activation_tanh_kernel(float *matrix, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        matrix[index] = tanh(matrix[index]);
    }
}

//flatten 
__global__ void flatten(float *input, float *output, int width, int height, int depth) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height * depth) {
        output[index] = input[index];
    }
}


void readMNIST(char* filename) {
    FILE *fptr;
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char pixel;

    if ((fptr = fopen(filename, "rb")) == NULL) {
        printf("Can't open file\n");
        exit(1);
    }

    // Lecture de l'entête du fichier
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    // Lecture des images
    for (int img = 0; img < NUM_IMAGES; img++) {
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                fread(&pixel, sizeof(unsigned char), 1, fptr);
                images[img][i * WIDTH + j] = pixel / 255.0f;
            }
        }
    }

    fclose(fptr);
}





int main() {
    srand(time(NULL));

    readMNIST("train-images.idx3-ubyte");

     // Allocation de mémoire sur le GPU
    float *d_C1_kernel, *d_C1_data, *d_S1_data, *d_C3_kernel, *d_C3_data, *d_S3_data, *d_flatten_data;
    float *d_C5_weights, *d_C5_data, *d_F6_weights, *d_F6_data, *d_output_weights, *d_output_data;



    cudaMalloc(&d_raw_data, WIDTH * HEIGHT * sizeof(float));

    cudaMalloc(&d_C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT * sizeof(float));
    cudaMalloc(&d_C3_kernel, C3_DEPTH * C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_C3_data, C3_DEPTH * C3_WIDTH * C3_HEIGHT * sizeof(float));
    cudaMalloc(&d_S3_data, C3_DEPTH * S3_WIDTH * S3_HEIGHT * sizeof(float));
    cudaMalloc(&d_flatten_data, FLATTEN_SIZE * sizeof(float));
    cudaMalloc(&d_C5_weights, FLATTEN_SIZE * C5_SIZE * sizeof(float));
    cudaMalloc(&d_C5_data, C5_SIZE * sizeof(float));
    cudaMalloc(&d_F6_weights, C5_SIZE * F6_SIZE * sizeof(float));
    cudaMalloc(&d_F6_data, F6_SIZE * sizeof(float));
    cudaMalloc(&d_output_weights, F6_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_data, OUTPUT_SIZE * sizeof(float));



    // Initialisation des poids et des données
    cudaMemcpy(d_raw_data, images[0], WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    initMatrix(C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE);
    cudaMemcpy(d_C1_kernel, C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    initMatrix(C3_kernel, C3_DEPTH * C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE);
    cudaMemcpy(d_C3_kernel, C3_kernel, C3_DEPTH * C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    initMatrix(C5_weights, FLATTEN_SIZE * C5_SIZE);
    cudaMemcpy(d_C5_weights, C5_weights, FLATTEN_SIZE * C5_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    initMatrix(F6_weights, C5_SIZE * F6_SIZE);
    cudaMemcpy(d_F6_weights, F6_weights, C5_SIZE * F6_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    initMatrix(output_weights, F6_SIZE * OUTPUT_SIZE);
    cudaMemcpy(d_output_weights, output_weights, F6_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);



    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks;

    // Convolution Layer 1
    numBlocks = dim3((C1_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (C1_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2D<<<numBlocks, threadsPerBlock>>>(d_raw_data, d_C1_data, d_C1_kernel, WIDTH, HEIGHT, KERNEL_SIZE, C1_DEPTH);

    // Pooling Layer 1
    numBlocks = dim3((S1_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (S1_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    subSample<<<numBlocks, threadsPerBlock>>>(d_C1_data, d_S1_data, C1_WIDTH, C1_HEIGHT, C1_DEPTH);

    // Convolution Layer 2
    numBlocks = dim3((C3_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (C3_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2D<<<numBlocks, threadsPerBlock>>>(d_S1_data, d_C3_data, d_C3_kernel, S1_WIDTH, S1_HEIGHT, KERNEL_SIZE, C3_DEPTH);

    // Pooling Layer 2
    numBlocks = dim3((S3_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (S3_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    subSample<<<numBlocks, threadsPerBlock>>>(d_C3_data, d_S3_data, C3_WIDTH, C3_HEIGHT, C3_DEPTH);

    // Flatten Layer
    numBlocks = dim3((FLATTEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);
    flatten<<<numBlocks, threadsPerBlock>>>(d_S3_data, d_flatten_data, S3_WIDTH, S3_HEIGHT, S3_DEPTH);

    // Fully Connected Layer 1 (Dense)
    numBlocks = dim3((FLATTEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);
    fullyConnectedLayer<<<numBlocks, threadsPerBlock>>>(d_flatten_data, d_C5_data, d_C5_weights, FLATTEN_SIZE, C5_SIZE);

    // Fully Connected Layer 2 (Dense)
    numBlocks = dim3((C5_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);
    fullyConnectedLayer<<<numBlocks, threadsPerBlock>>>(d_C5_data, d_F6_data, d_F6_weights, C5_SIZE, F6_SIZE);

    // Output Layer (Dense)
    numBlocks = dim3((F6_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x);
    fullyConnectedLayer<<<numBlocks, threadsPerBlock>>>(d_F6_data, d_output_data, d_output_weights, F6_SIZE, OUTPUT_SIZE);

    // Libération de la mémoire GPU pour les autres tableaux
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_C3_kernel);
    cudaFree(d_C3_data);
    cudaFree(d_S3_data);
    cudaFree(d_flatten_data);
    cudaFree(d_C5_weights);
    cudaFree(d_C5_data);
    cudaFree(d_F6_weights);
    cudaFree(d_F6_data);
    cudaFree(d_output_weights);
    cudaFree(d_output_data);



    return 0;
}
