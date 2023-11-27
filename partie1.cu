#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Initialisation de la matrice
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
    
}