#include <cstdio>
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>

#include "device_launch_parameters.h"
#include "gpuTimer.h"
#include "matplotlibcpp.h"

#define cudaCheckError() {                                  \
    cudaError_t e = cudaGetLastError();                     \
    if (e != cudaSuccess) {                                 \
        printf("CUDA error %s %d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(e));     \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}                                                           \

namespace plt = matplotlibcpp;

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    auto free_mb = static_cast<float>(free)/1024.f/1024.f;
    auto total_mb = static_cast<float>(total)/1024.f/1024.f;
    std::cout << "Device memory (free/total) = " << free_mb << "/" << total_mb << "MB\n" << std::endl;
    std::cout << "Used memory = " << total_mb - free_mb << "MB\n" << std::endl;
}

void setGrid(
    std::vector<float>& x
    , std::vector<float>& k
    , float dx, float dk
    , const int N
    )
{
    for (auto i = 0; i < N; ++i) {
        x[i] = (i - N / 2) * dx;
        k[i] = (i - N / 2) * dk; 
    }
}

void generateGaussian(
    std::vector<float>& data
    , float x0 // shift
    , float dx
    , float sigma
    , float coeff = 1.f
    ) 
{
    int N = data.size();
    for (auto i = 0; i < N; ++i) {
        auto x = (i - N / 2) * dx;
        data[i] = coeff * expf(- (x - x0) * (x - x0) / (2.f * sigma * sigma));
    }
}

void generateAnswer(
    std::vector<float>& real  
    , std::vector<float>& imag   
    , float x0
    , float dk 
    , float sigma
    , float coeff
    )
{
    int N = real.size(); 
    for (auto i = 0; i < N; ++i) {
        auto k = (i - N / 2) * dk;
        auto envelope = coeff * expf(- 0.5f * k * k * sigma * sigma);
        real[i] = envelope * cosf(-k*x0);
        imag[i] = envelope * sinf(-k*x0);
    }
}

__global__ void toComplex(const float* data, cufftComplex* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i].x = data[i];
        out[i].y = 0.f;
    }
}

template<typename T>
__global__ void fftshift(T* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N / 2) {
        T tmp = data[i];
        data[i] = data[i + N / 2];
        data[i + N / 2] = tmp;
    }
}

template<typename T>
__global__ void scale(T* input, T scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        input[i] = scale * input[i];
    }
}

template<>
__global__ void scale<cufftComplex>(cufftComplex* input, cufftComplex scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        input[i] = cuCmulf(input[i], scale);
    }
}

// in-place
template<typename T>
void api_fftshift(T* input, const int size, cudaStream_t stream) {
    int block_size = 128;
    int grid_shift = (size / 2 + block_size - 1) / block_size;
    fftshift<<<grid_shift, block_size, 0, stream>>>(input, size); // which one is faster? thrust? or this?
}

// in-place
template<typename T>
void api_scale(T* input, T scale_coeff, const int size, cudaStream_t stream) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    scale<<<grid_size, block_size, 0, stream>>>(input, scale_coeff, size);
}

// out-of-place
void api_makeComplex(const float* in, cufftComplex* out, const int size, cudaStream_t stream) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    toComplex<<<grid_size, block_size, 0, stream>>>(in, out, size);
}
 
// out-of-place
void api_fftC2C(cufftComplex* in, cufftComplex* out, const int size, int direction, cudaStream_t stream) {
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2C, 1);
    cufftSetStream(plan, stream);
    cufftExecC2C(plan, in, out, direction);
    cufftDestroy(plan);
} 

// out-of-place
void api_fftR2C(float* in, cufftComplex* out, const int size, int direction, cudaStream_t stream) {
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, 1);
    cufftSetStream(plan, stream);
    cufftExecR2C(plan, in, out); // R2C has no direction
    cufftDestroy(plan);
}

constexpr double pi = 3.141592655358979323846;

int main(void)
{
    // data preparation 
    const int N = 128;
    const float dx = 8.f;
    const float sigma = 100.f;
    const float width = dx * N;
    const float dk = 2.f * pi / (dx * N);
    const float x0 = 14.f * dx;

    auto h_x = std::vector<float>(N);
    auto h_k = std::vector<float>(N);
    setGrid(h_x, h_k, dx, dk, N);

    auto h_input = std::vector<float>(N);
    generateGaussian(h_input, x0, dx, sigma);

    // plot
    plt::plot(h_x, h_input);
    plt::show();
    
    // cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // gpu memory allocation
    float* d_input = nullptr;
    cufftComplex* d_cplx_input = nullptr;
    cufftComplex* d_output = nullptr;
    cufftComplex* d_output_r2c = nullptr;

    cudaMallocAsync(&d_input, sizeof(float) * N, stream);
    cudaMallocAsync(&d_cplx_input, sizeof(cufftComplex) * N, stream);
    cudaMallocAsync(&d_output, sizeof(cufftComplex) * N, stream);
    cudaMallocAsync(&d_output_r2c, sizeof(cufftComplex) * (N/2 + 1), stream);
    cudaMemcpyAsync(d_input, h_input.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream);

    GpuTimer timer("main algorithm");
    timer.start();

    // fftshift
    api_fftshift<float>(d_input, N, stream);
    
    // make complex
    api_makeComplex(d_input, d_cplx_input, N, stream);
    cudaFreeAsync(d_input, stream);    

    // fft
    api_fftC2C(d_cplx_input, d_output, N, CUFFT_FORWARD, stream);
    cudaFreeAsync(d_cplx_input, stream);    

    // fft r2c test
    api_fftR2C(d_input, d_output_r2c, N, CUFFT_FORWARD, stream);

    // fftshift
    api_fftshift<cufftComplex>(d_output, N, stream); 

    // scale
    api_scale<cufftComplex>(d_output, make_cuComplex(dx, 0.f), N, stream);
    api_scale<cufftComplex>(d_output_r2c, make_cuComplex(dx, 0.f), N / 2 + 1, stream);

    checkDeviceMemory();
    timer.stop();
    timer.printElapsedTime();

    cudaStreamSynchronize(stream);

    // copy to host
    std::vector<cufftComplex> h_fk(N);
    cudaMemcpyAsync(h_fk.data(), d_output, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost, stream); 

    std::vector<cufftComplex> h_fk_r2c(N/2 + 1);
    cudaMemcpyAsync(h_fk_r2c.data(), d_output_r2c, sizeof(cufftComplex) * (N / 2 + 1), cudaMemcpyDeviceToHost, stream); 
    
    // accuracy test
    // cplx_host to real_host
    std::vector<float> h_fk_real(N);
    std::vector<float> h_fk_imag(N);
    for (auto i = 0u; i < h_fk.size(); ++i) {
        h_fk_real[i] = h_fk[i].x;
        h_fk_imag[i] = h_fk[i].y;

        std::cout << "h_fk[" << i << "] " << h_fk_real[i] << " + i " << h_fk_imag[i] << std::endl;
    }

    for (auto i = 0u; i < h_fk_r2c.size(); ++i) {
        std::cout << "h_fk_r2c[" << i << "] " << h_fk_r2c[i].x << " + i " << h_fk_r2c[i].y << std::endl;
    }

    std::vector<float> h_fk_real_theory(N);
    std::vector<float> h_fk_imag_theory(N);
    generateAnswer(
        h_fk_real_theory
        , h_fk_imag_theory
        , x0, dk, sigma
        , sigma * sqrt(2.f * pi)
        );

    for (auto i = 0u; i < N; ++i) {
        if ((abs(h_fk_real[i] - h_fk_real_theory[i]) > 0.1)
            or (abs(h_fk_imag[i] - h_fk_imag_theory[i]) > 0.1)
            ) 
        {
            std::cout << "large numerical differences! test failed" << std::endl;
        }
    }
    std::cout << "accuracy test: success" << std::endl;
    
//    // plot
//    plt::plot(h_k, h_fk_real);
//    plt::show();

    // clean up
    cudaFreeAsync(d_output, stream);    

    return 0;
}

