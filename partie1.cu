#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
void MatrixMult(float *M1, float *M2, float *Mout, int n);

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n);



void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        M[i] = ((float)rand() / (float)(RAND_MAX)) * 2.0 - 1.0;
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", M[i * p + j]);
        }
        printf("\n");
    }
}


void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n * p; i++) {
        Mout[i] = M1[i] + M2[i];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = sum;
        }
    }
}


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < p) {
        Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
    }
}


__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += M1[i * n + k] * M2[k * n + j];
        }
        Mout[i * n + j] = sum;
    }
}


int main(){
    int n=100;
    int p=100;
    int size = n * p * sizeof(float);


    float *h_M1 = (float *)malloc(size);
    float *h_M2 = (float *)malloc(size);
    float *h_Mout = (float *)malloc(size);
    MatrixInit(h_M1, n, p);
    MatrixInit(h_M2, n, p);
    MatrixInit(h_Mout, n, p);


    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void **)&d_M1, size);
    cudaMalloc((void **)&d_M2, size);
    cudaMalloc((void **)&d_Mout, size);


    cudaMemcpy(d_M1, h_M1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, h_M2, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (p + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaMatrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n, p);

    cudaMemcpy(h_Mout, d_Mout, size, cudaMemcpyDeviceToHost);

    MatrixPrint(h_Mout, n, p);

    free(h_M1);
    free(h_M2);
    free(h_Mout);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;

}

