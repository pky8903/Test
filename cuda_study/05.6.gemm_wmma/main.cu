#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mma.h>
#include <cuda_fp16.h>

#define DATA_TYPE_HALF __half
#define DATA_TYPE float

#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define INDEX2ROW(_index,_width)	(int)((_index)/(_width))
#define INDEX2COL(_index,_width)	((_index)%(_width))
#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))

#define BLOCK_SIZE 4

// macro function
#define IS_EQUAL(_a, _b) (abs(_b - _a) < 10e-2)

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

/******************************************************************
* Modify this kernel to use TensorCore 
******************************************************************/
__global__ void MatMul_TensorCore(
    DATA_TYPE_HALF* a
    , DATA_TYPE_HALF* b
    , DATA_TYPE* c
    , const int M
    , const int N 
    , const int K 
    )
{
    int lda = M;
    int ldb = K;
    int ldc = M;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    wmma::fragment<
        wmma::matrix_a
        , WMMA_M
        , WMMA_N
        , WMMA_K
        , DATA_TYPE_HALF
        , wmma::row_major
        > a_frag;

    wmma::fragment<
        wmma::matrix_b
        , WMMA_M
        , WMMA_N
        , WMMA_K
        , DATA_TYPE_HALF
        , wmma::row_major
        > b_frag;
    
    wmma::fragment<
        wmma::accumulator
        , WMMA_M
        , WMMA_N
        , WMMA_K
        , DATA_TYPE
        > acc_frag;

    wmma::fragment<
        wmma::accumulator
        , WMMA_M
        , WMMA_N
        , WMMA_K
        , DATA_TYPE
        > c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // loop over k
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // bound checking
        if (aRow < M and aCol < K and bRow < K and bCol < N) {
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
 
    if (cRow < M && cCol < N) {
//        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);
        wmma::store_matrix_sync(c + cRow * ldc + cCol, acc_frag, ldc, wmma::mem_row_major);
    }
}

void floatToHalf(const DATA_TYPE* src, DATA_TYPE_HALF* dst, size_t size){
    for (size_t i = 0; i < size; ++i) {
        dst[i] = __float2half(src[i]);
    }
}
 
__global__ void MatMul_SharedMem(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
    __shared__ DATA_TYPE subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DATA_TYPE subB[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    DATA_TYPE val = 0;

    for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); ++bID) {
        int offset = bID * BLOCK_SIZE;
        
        if (row >= m or offset + localCol >= k) {
            subA[localCol][localRow] = 0;
        }
        else {
            subA[localCol][localRow] = matA[row * k + (offset + localCol)];
        }

        if (col >= n or offset + localRow >= k) {
            subB[localRow][localCol] = 0;
        }
        else {
            subB[localRow][localCol] = matB[(offset + localRow) * n + col];
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            val += subA[i][localRow] * subB[i][localCol];
        }

        __syncthreads();
    }

	if (row >= m || col >= n) { return; }

	matC[row * n + col] = val;
}
/******************************************************************
******************************************************************/

template<class T> void allocNinitMem(T** p, long long size, DATA_TYPE* memUsage = NULL);
void runMatMul_Basic(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k);
bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size);

DS_timer timer(10);
void setTimer();

int main(int argc, char* argv[])
{
	setTimer();

	// set matrix size
	int m, n, k;
	m = SIZE_M;
	n = SIZE_N;
	k = SIZE_K;

	printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

	int sizeA = m * k;
	int sizeB = k * n;
	int sizeC = m * n;

	// Make matrix
	DATA_TYPE* A = NULL, * B = NULL;
	allocNinitMem<DATA_TYPE>(&A, sizeA);
	allocNinitMem<DATA_TYPE>(&B, sizeB);

	DATA_TYPE* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<DATA_TYPE>(&Ccpu, sizeC);
	allocNinitMem<DATA_TYPE>(&Cgpu, sizeC);

	// generate input matrices
	for (int i = 0; i < sizeA; i++) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	for (int i = 0; i < sizeB; i++) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

    auto A_half = (DATA_TYPE_HALF*)malloc(sizeA * sizeof(DATA_TYPE_HALF));
    auto B_half = (DATA_TYPE_HALF*)malloc(sizeB * sizeof(DATA_TYPE_HALF));

    floatToHalf(A, A_half, sizeA); 
    floatToHalf(B, B_half, sizeB); 

	// CPU version (OpenMP)
	timer.onTimer(0);
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int cIndex = row * n + col;
			Ccpu[cIndex] = 0;
			for (int i = 0; i < k; i++)
				Ccpu[cIndex] += (A[row * k + i] * B[i * n + col]);
		}
	}
	printf("CPU finished!\n");
	timer.offTimer(0);

	// GPU setup
	DATA_TYPE_HALF* dA, * dB;
    DATA_TYPE* dC;

	cudaMalloc(&dA, sizeA * sizeof(DATA_TYPE_HALF));
	cudaMemset(dA, 0, sizeA * sizeof(DATA_TYPE_HALF));

	cudaMalloc(&dB, sizeB * sizeof(DATA_TYPE_HALF));
	cudaMemset(dB, 0, sizeB * sizeof(DATA_TYPE_HALF));

	cudaMalloc(&dC, sizeC * sizeof(DATA_TYPE));
	cudaMemset(dC, 0, sizeC * sizeof(DATA_TYPE));

	timer.onTimer(1);

	timer.onTimer(4);
	cudaMemcpy(dA, A_half, sizeA * sizeof(DATA_TYPE_HALF), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B_half, sizeB * sizeof(DATA_TYPE_HALF), cudaMemcpyHostToDevice);
	timer.offTimer(4);

	/******************************************************************
	* Write your codes for GPU algorithm from here
	******************************************************************/
	// Sharead memroy version

	// 1. set the thread layout
	// Change the layout if you need
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (SIZE_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (SIZE_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	// 2. kernel call
	timer.onTimer(3);
    MatMul_TensorCore << <gridDim, blockDim >> > (dA, dB, dC, m, n, k); 
//	MatMul_SharedMem << <gridDim, blockDim >> > (dA, dB, dC, m, n, k);
	cudaDeviceSynchronize();
	timer.offTimer(3);

	/******************************************************************
	******************************************************************/

	timer.onTimer(5);
	cudaMemcpy(Cgpu, dC, sizeC * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	timer.offTimer(5);

	timer.offTimer(1);

	// Basci version
//	runMatMul_Basic(dA, dB, dC, m, n, k);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	printf("[Kernel (Tensor core)] ");
//	compareMatrix(Ccpu, Cgpu, sizeC);

	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}

bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size)
{
	bool isMatched = true;
	for (int i = 0; i < _size; i++) {
		if (!IS_EQUAL(_A[i], _B[i])) {
			printf("[%d] not matched! (%f, %f)\n", i, _A[i], _B[i]);
			getchar();
			isMatched = false;
		}
	}
	if (isMatched)
		printf("Results are matched!\n");
	else
		printf("Results are not matched!!!!!!!!!!!\n");

	return isMatched;
}

__global__ void MatMul(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row >= m || col >= n)
		return;

	DATA_TYPE val = 0; // hope to use register
	for (int i = 0; i < k; i++)
		val += matA[ID2INDEX(row, i, k)] * matB[ID2INDEX(i, col, n)];

	matC[ID2INDEX(row, col, n)] = val;
}

void runMatMul_Basic(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	dim3 gridDim(ceil((float)n / BLOCK_SIZE), ceil((float)m / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	timer.onTimer(7);
	MatMul << < gridDim, blockDim >> > (matA, matB, matC, m, n, k);
	cudaDeviceSynchronize();
	timer.offTimer(7);

	cudaMemset(matC, 0, m * n * sizeof(DATA_TYPE));
}

template<class T>
void allocNinitMem(T** p, long long size, DATA_TYPE* memUsage) {
	*p = new T[size];
	memset(*p, 0, sizeof(T) * size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T) * size;
	}
}

void setTimer()
{
	timer.setTimerName(0, (char*)"CPU algorithm");
	timer.setTimerName(1, (char*)"GPU/CUDA algorithm");
	timer.setTimerName(3, (char*)" - Kernel (Tensor core)");
	timer.setTimerName(4, (char*)" - [Data transter] host->device");
	timer.setTimerName(5, (char*)" - [Data transfer] device->host");
	timer.setTimerName(7, (char*)"Kernel (Basic)");
}
