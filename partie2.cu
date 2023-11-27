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


initRandomMatrix(raw_data, WIDTH * HEIGHT);
initMatrix(C1_data, C1_DEPTH * C1_WIDTH * C1_HEIGHT, 0);
initMatrix(S1_data, C1_DEPTH * S1_WIDTH * S1_HEIGHT, 0);
initRandomMatrix(C1_kernel, C1_DEPTH * KERNEL_SIZE * KERNEL_SIZE);



