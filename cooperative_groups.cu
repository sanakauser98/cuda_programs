#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// CUDA kernel using cooperative groups
__global__ void printPattern() {
    // Create a thread block group
    cg::thread_block block = cg::this_thread_block();

    int row = block.thread_index().x + 1; // Row number (starting from 1)
    int numElements = 6 - row; // Number of elements in each row

    if (row <= 5) {
        for (int i = 0; i < numElements; i++) {
            printf("%d ", row);
        }
        printf("\n");

        // Synchronize within the block to maintain print order
        block.sync();
    }
}

int main() {
    // Launch kernel with 1 block of 5 threads
    printPattern<<<1, 5>>>();

    // Synchronize to ensure output before program exits
    cudaDeviceSynchronize();

    return 0;
}
