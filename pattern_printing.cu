#include <stdio.h>

// CUDA kernel to print the pattern
__global__ void printPattern() {
    int row = threadIdx.x + 1; // Row number (starting from 1)
    int numElements = 6 - row; // Number of elements in each row

    if (row <= 5) { // Ensure we process only 5 rows
        for (int i = 0; i < numElements; i++) {
            printf("%d ", row);
        }
        printf("\n"); // New line after printing each row
    }
}

int main() {
    // Launch kernel with 5 threads (one for each row)
    printPattern<<<1, 5>>>();

    // Synchronize to ensure output before program exits
    cudaDeviceSynchronize();

    return 0;
}
