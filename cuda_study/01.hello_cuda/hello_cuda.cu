#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define cudaCheckError() {                                  \
    cudaError_t e = cudaGetLastError();                     \
    if (e != cudaSuccess) {                                 \
        printf("CUDA error %s %d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(e));     \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}                                                           \

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory (free/total) = %lld/%lld bytes\n", free, total);
}

__global__ void helloCUDA(void)
{
    printf("Hello CUDA from GPU!\n");
}

int main(void)
{
    printf("Hello GPU from CPU!\n");
    helloCUDA<<<2, 100>>>();

    cudaDeviceSynchronize();

    int* dDataPtr;
    cudaError_t errorCode;

    checkDeviceMemory();
    errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
    printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode));
    checkDeviceMemory();

    cudaDeviceSynchronize();
 
    return 0;
}

