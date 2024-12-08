#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.h"

// The size of the vector
#define NUM_DATA 1700000

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAdd(int* _a, int* _b, int* _c) {
	int tID = blockDim.x * blockIdx.x + threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    // set timer
    DS_timer timer(5);
    timer.setTimerName(0, "CUDA_Total");
    timer.setTimerName(1, "Computation(Kernel)");
    timer.setTimerName(2, "Data Trans.: Host -> Device");
    timer.setTimerName(3, "Data Trans.: Device -> Host");
    timer.setTimerName(4, "VecAdd On Host");
    timer.initTimers();

	int* a, * b, * c, * h_c;	// Vectors on the host
	int* d_a, * d_b, * d_c;		// Vectors on the device

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

	// Memory allocation on the host-side
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	h_c = new int[NUM_DATA]; memset(h_c, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparision)
    timer.onTimer(4);
	for (int i = 0; i < NUM_DATA; i++)
		h_c[i] = a[i] + b[i];
    timer.offTimer(4);

	//****************************************//
	//******* Write your code - start ********//

	// 1. Memory allocation on the device-side (d_a, d_b, d_c)
    cudaMalloc(&d_a, memSize); cudaMemset(d_a, 0, memSize);
    cudaMalloc(&d_b, memSize); cudaMemset(d_b, 0, memSize);
    cudaMalloc(&d_c, memSize); cudaMemset(d_c, 0, memSize);

    timer.onTimer(0);

	// 2. Data copy : Host (a, b) -> Device (d_a, d_b)
    timer.onTimer(2);
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(2);

	// 3. Kernel call
    timer.onTimer(1);
    dim3 dimGrid(ceil((float)NUM_DATA/1024), 1, 1);
    dim3 dimBlock(1024, 1, 1);
	vecAdd <<<dimGrid, dimBlock >>> (d_a, d_b, d_c);
    timer.offTimer(1);

	// 4. Copy results : Device (d_c) -> Host (c)
    timer.onTimer(3);
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    timer.offTimer(3);
    cudaDeviceSynchronize();

    timer.offTimer(0);

	// 5. Release device memory (d_a, d_b, d_c)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    timer.printTimer();

	//******** Write your code - end *********//
	//****************************************//

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if (h_c[i] != c[i]) {
			printf("[%d] The resutl is not matched! (%d, %d)\n"
				, i, h_c[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	// Release host memory
	delete[] a; delete[] b; delete[] c;

	return 0;
}
