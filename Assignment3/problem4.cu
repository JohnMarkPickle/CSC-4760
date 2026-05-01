//problem4.cu
//
//how to run
//
//nvcc -o p4 problem4.cu
//./p3

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 3
#define COLS 3
#define N (ROWS * COLS)

//error checks
#define CUDA_CHECK(call)                                                        
    do {                                                                        
        cudaError_t err = (call);                                               
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA error at %s:%d — %s\n",
                    __FILE__, __LINE__, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }                                                                       
    } while (0)

__global__ void rowwise_add(int *A, const int *B, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= rows * cols)
        return;

    int col = idx % cols;
    A[idx] += B[col];
}

//print matrix
static void print_matrix(const char *label, const int *M, int rows, int cols)
{
    printf("%s\n", label);
    for (int r = 0; r < rows; r++) 
    {
        printf("  [");
        for (int c = 0; c < cols; c++) 
        {
            printf("%4d", M[r * cols + c]);
            if (c < cols - 1) printf("  ");
        }
        printf(" ]\n");
    }
    printf("\n");
}

int main(void)
{

    int h_A[N] = {130, 147, 115, 224, 158, 187, 54, 158, 120};

    int h_B[COLS] = {221, 12, 157};

    int expected[N] = {351, 159, 272, 445, 170, 344, 275, 170, 277};

    print_matrix("Matrix A:", h_A, ROWS, COLS);
    printf("Vector B: [%d  %d  %d]\n\n", h_B[0], h_B[1], h_B[2]);

    int *d_A, *d_B;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N     * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, COLS  * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N    * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, COLS * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 9;
    int gridSize = (N + blockSize - 1) / blockSize;

    printf("Kernel launch: gridSize=%d  blockSize=%d  total threads=%d\n\n", gridSize, blockSize, gridSize * blockSize);
    rowwise_add<<<gridSize, blockSize>>>(d_A, d_B, ROWS, COLS);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result[N];
    CUDA_CHECK(cudaMemcpy(h_result, d_A, N * sizeof(int), cudaMemcpyDeviceToHost));

    print_matrix("Result:", h_result, ROWS, COLS);
    print_matrix("Expected:", expected, ROWS, COLS);

    int pass = 1;
    for (int i = 0; i < N; i++) 
    {
        if (h_result[i] != expected[i]) 
        {
            printf("MISMATCH %d (row=%d col=%d): "
                   "got %d, expected %d\n",
                   i, i / COLS, i % COLS, h_result[i], expected[i]);
            pass = 0;
        }
    }

    if (pass)
        printf("Correct! — all %d elements match.\n", N);
    else
        printf("Wrong! — see mismatches above.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
