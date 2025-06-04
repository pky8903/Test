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

constexpr double pi = 3.141592655358979323846;

namespace plt = matplotlibcpp;

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    auto free_mb = static_cast<float>(free)/1024.f/1024.f;
    auto total_mb = static_cast<float>(total)/1024.f/1024.f;
    std::cout << "Device memory (free/total) = " << free_mb << "/" << total_mb << " MB" << std::endl;
    std::cout << "Used memory = " << total_mb - free_mb << " MB" << std::endl;
}

int parse_option(int argc, char* argv[], const std::string& key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == key) {
            try {
                return std::stoi(argv[i + 1]);
            }
            catch (const std::exception& e) {
                std::cerr << "Invalid value for " << key << ": " << argv[i + 1] << std::endl;
                std::exit(1);
            }
        }
    }
    
    std::cerr << "Invalid value for " << key << std::endl;
    std::exit(1);
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

__global__ void toReal(const cufftComplex* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = in[i].x;
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

// input1 = input1 * input2
template<typename T>
__global__ void mul(T* input1, T* input2, int N);

template<>
__global__ void mul<float>(float* input1, float* input2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        input1[i] = input1[i] * input2[i]; 
    }
}
    
template<>
__global__ void mul<cufftComplex>(cufftComplex* input1, cufftComplex* input2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        input1[i] = cuCmulf(input1[i], input2[i]); 
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

// in-place
template<typename T>
void api_mul(T* input1, T* input2, const int size, cudaStream_t stream) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    mul<<<grid_size, block_size, 0, stream>>>(input1, input2, size);
}

// out-of-place
void api_makeComplex(const float* in, cufftComplex* out, const int size, cudaStream_t stream) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    toComplex<<<grid_size, block_size, 0, stream>>>(in, out, size);
}

// out-of-place
void api_makeReal(const cufftComplex* in, float* out, const int size, cudaStream_t stream) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    toReal<<<grid_size, block_size, 0, stream>>>(in, out, size);
}
 
// out-of-place
void api_fftC2C(cufftComplex* in, cufftComplex* out, const int size, int direction, cudaStream_t stream) {
    GpuTimer timer("api_fftC2C");
    timer.start();
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2C, 1);
    cufftSetStream(plan, stream);
    cufftExecC2C(plan, in, out, direction);
    cufftDestroy(plan);

    cudaStreamSynchronize(stream);
    timer.stop();
    timer.printElapsedTime();
} 

// out-of-place
void api_fftR2C(float* in, cufftComplex* out, const int size, int direction, cudaStream_t stream) {
    GpuTimer timer("api_fftR2C");
    timer.start();
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, 1);
    cufftSetStream(plan, stream);
    cufftExecR2C(plan, in, out); // R2C has no direction

    cudaStreamSynchronize(stream);
    cufftDestroy(plan);
    timer.stop();
    timer.printElapsedTime();
}

// out-of-place
void api_fftC2R(cufftComplex* in, float* out, const int size, int direction, cudaStream_t stream) {
    GpuTimer timer("api_fftC2R");
    timer.start();
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2R, 1);
    cufftSetStream(plan, stream);
    cufftExecC2R(plan, in, out); // C2R has no direction

    cudaStreamSynchronize(stream);
    cufftDestroy(plan);
    timer.stop();
    timer.printElapsedTime();
}

void fft_C2C(float* d_input
    , const int size
    , const float dx
    , int direction
    , cudaStream_t stream
    , cufftComplex* d_output
    )  
{
    cufftComplex* d_cplx_input = nullptr;
    cudaMallocAsync(&d_cplx_input, sizeof(cufftComplex) * size, stream);

    // fftshift
    api_fftshift<float>(d_input, size, stream);
    
    // make complex
    api_makeComplex(d_input, d_cplx_input, size, stream);
    cudaFreeAsync(d_input, stream);    
    
    // fft
    api_fftC2C(d_cplx_input, d_output, size, direction, stream);

    // free buffer
    cudaFreeAsync(d_cplx_input, stream);    

    // fftshift
    api_fftshift<cufftComplex>(d_output, size, stream); 

    // scale
    api_scale<cufftComplex>(d_output, make_cuComplex(dx, 0.f), size, stream);
}

void conv_C2C(float* d_input
    , float* d_kernel
    , const int size
    , const float dx
    , cudaStream_t stream
    , float* d_output
    )
{
    cufftComplex* d_cplx_input = nullptr;
    cudaMallocAsync(&d_cplx_input, sizeof(cufftComplex) * size, stream);

    cufftComplex* d_cplx_kernel = nullptr;
    cudaMallocAsync(&d_cplx_kernel, sizeof(cufftComplex) * size, stream);

    cufftComplex* d_ffted_input = nullptr;
    cudaMallocAsync(&d_ffted_input, sizeof(cufftComplex) * size, stream);

    cufftComplex* d_ffted_kernel = nullptr;
    cudaMallocAsync(&d_ffted_kernel, sizeof(cufftComplex) * size, stream);

    cufftComplex* d_cplx_output = nullptr;
    cudaMallocAsync(&d_cplx_output, sizeof(cufftComplex) * size, stream);

    // fftshift
    api_fftshift<float>(d_input, size, stream);
    api_fftshift<float>(d_kernel, size, stream);

    // make_complex
    api_makeComplex(d_input, d_cplx_input, size, stream);
    api_makeComplex(d_kernel, d_cplx_kernel, size, stream);

    // fft
    api_fftC2C(d_cplx_input, d_ffted_input, size, CUFFT_FORWARD, stream);
    api_fftC2C(d_cplx_kernel, d_ffted_kernel, size, CUFFT_FORWARD, stream);
    cudaFreeAsync(d_cplx_input, stream);
    cudaFreeAsync(d_cplx_kernel, stream);

    // point-wise multiplication
    api_mul(d_ffted_input, d_ffted_kernel, size, stream); 
    auto& d_ffted_output = d_ffted_input;
    cudaFreeAsync(d_ffted_kernel, stream);
    
    // ifft
    api_fftC2C(d_ffted_output, d_cplx_output, size, CUFFT_INVERSE, stream);

    // to real
    api_makeReal(d_cplx_output, d_output, size, stream);
    cudaFreeAsync(d_cplx_output, stream);

    // ifftshift
    api_fftshift<float>(d_output, size, stream);

    // scale
    api_scale<float>(d_output, dx/(static_cast<float>(size)), size, stream);
}

void conv_R2C(float* d_input
    , float* d_kernel
    , const int size
    , const float dx
    , cudaStream_t stream
    , float* d_output
    )
{
    cufftComplex* d_ffted_input = nullptr;
    cudaMallocAsync(&d_ffted_input, sizeof(cufftComplex) * (size / 2 + 1), stream);

    cufftComplex* d_ffted_kernel = nullptr;
    cudaMallocAsync(&d_ffted_kernel, sizeof(cufftComplex) * (size / 2 + 1), stream);
    
    // fftshift
    api_fftshift<float>(d_input, size, stream);
    api_fftshift<float>(d_kernel, size, stream);

    // fft
    api_fftR2C(d_input, d_ffted_input, size, CUFFT_FORWARD, stream);
    api_fftR2C(d_kernel, d_ffted_kernel, size, CUFFT_FORWARD, stream);

    // point-wise multiplicataion
    api_mul(d_ffted_input, d_ffted_kernel, size, stream);
    auto& d_ffted_output = d_ffted_input;

    // ifft
    api_fftC2R(d_ffted_output, d_output, size, CUFFT_INVERSE, stream);
    
    // fftshift
    api_fftshift<float>(d_output, size, stream);
    
    // scale
    api_scale<float>(d_output, dx/(static_cast<float>(size)), size, stream);
}

void fft_R2C(float* d_input
    , const int size
    , const float dx
    , int direction
    , cudaStream_t stream    
    , cufftComplex* d_output_r2c
    )  
{
    // fftshift
    api_fftshift<float>(d_input, size, stream);

    // fft r2c test
    api_fftR2C(d_input, d_output_r2c, size, direction, stream);

    // scale
    api_scale<cufftComplex>(d_output_r2c, make_cuComplex(dx, 0.f), size / 2 + 1, stream);
}

void test_C2C(
    cufftComplex* d_output
    , const int N
    , const float x0
    , const float dx
    , const float dk
    , const float sigma
    , cudaStream_t stream
    )
{
    std::vector<cufftComplex> h_fk(N);
    cudaMemcpyAsync(h_fk.data(), d_output, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost, stream); 

    // cplx_host to real_host
    std::vector<float> h_fk_real(N);
    std::vector<float> h_fk_imag(N);
    for (auto i = 0u; i < h_fk.size(); ++i) {
        h_fk_real[i] = h_fk[i].x;
        h_fk_imag[i] = h_fk[i].y;
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
}

void test_R2C(
    cufftComplex* d_output_r2c
    , const int N
    , const float x0
    , const float dx
    , const float dk
    , const float sigma
    , cudaStream_t stream
    )
{
    std::vector<cufftComplex> h_fk_r2c(N/2 + 1);
    cudaMemcpyAsync(h_fk_r2c.data(), d_output_r2c, sizeof(cufftComplex) * (N / 2 + 1), cudaMemcpyDeviceToHost, stream); 

    // cplx_host to real_host
    std::vector<float> h_fk_real(N/2 + 1);
    std::vector<float> h_fk_imag(N/2 + 1);
    for (auto i = 0u; i < h_fk_r2c.size(); ++i) {
        h_fk_real[i] = h_fk_r2c[i].x;
        h_fk_imag[i] = h_fk_r2c[i].y;
    }

    std::vector<float> h_fk_real_theory(N);
    std::vector<float> h_fk_imag_theory(N);
    generateAnswer(
        h_fk_real_theory
        , h_fk_imag_theory
        , x0, dk, sigma
        , sigma * sqrt(2.f * pi)
        );

    for (auto i = 0u; i < N/2 + 1; ++i) {
        if (i == N/2 + 1) {
            if ((abs(h_fk_real[i] - h_fk_real_theory[0]) > 0.1) or (abs(h_fk_imag[i] - h_fk_imag_theory[0]) > 0.1)) {
                std::cout << "large numerical differences! test failed" << std::endl;
            }
        }
        else {
            if ((abs(h_fk_real[i] - h_fk_real_theory[i + N/2]) > 0.1)
                or (abs(h_fk_imag[i] - h_fk_imag_theory[i + N/2]) > 0.1)
                ) 
            {
                std::cout << "large numerical differences! test failed" << std::endl;
            }

        }
    }
    std::cout << "accuracy test: success" << std::endl;
}

int main(int argc, char* argv[])
{
    // data preparation 
    const int N = parse_option(argc, argv, "--width");
    const int mode = parse_option(argc, argv, "--mode");
    const float dx = 8.f;
    const float sigma = 100.f;
    const float sigma_k = 50.f;
    const float width = dx * N;
    const float dk = 2.f * pi / (dx * N);
    const float x0 = 14.f * dx;

    auto h_x = std::vector<float>(N);
    auto h_k = std::vector<float>(N);
    setGrid(h_x, h_k, dx, dk, N);
    
    auto normalize_factor = [](const float sigma) {
        return 1.f/(sqrt(2.f * pi * sigma * sigma));
    };

    auto h_input = std::vector<float>(N);
    generateGaussian(h_input, x0, dx, sigma, normalize_factor(sigma));

    auto h_kernel = std::vector<float>(N);
    generateGaussian(h_kernel, 0.f, dx, sigma_k, normalize_factor(sigma_k));

    auto sigma_o = sqrt(sigma * sigma + sigma_k * sigma_k);
    auto h_output_theory = std::vector<float>(N);
    generateGaussian(h_output_theory, x0, dx, sigma_o, normalize_factor(sigma_o));

    // plot
    plt::plot(h_x, h_input);
    plt::show();
    
    // cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // gpu memory allocation
    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    cudaMallocAsync(&d_input, sizeof(float) * N, stream);
    cudaMallocAsync(&d_kernel, sizeof(float) * N, stream);
    cudaMallocAsync(&d_output, sizeof(float) * N, stream);

    cudaMemcpyAsync(d_input, h_input.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kernel, h_kernel.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream);

    // for GPU warming up
    std::cout << "GPU warming up - start" << std::endl;
    conv_C2C(d_input
        , d_kernel
        , N
        , dx
        , stream
        , d_output
        );
    conv_R2C(d_input
        , d_kernel
        , N
        , dx
        , stream
        , d_output
        );
    std::cout << "GPU warming up - end" << std::endl;

    GpuTimer timer("main_algorithm_" + std::to_string(N) + "_" + std::to_string(mode));
    timer.start();

    auto direction = CUFFT_FORWARD;
    switch (mode) {
        case 0:
            conv_C2C(d_input
                , d_kernel
                , N
                , dx
                , stream
                , d_output
                );
            break;
        case 1:
            conv_R2C(d_input
                , d_kernel
                , N
                , dx
                , stream
                , d_output
                );
            break;
        default:
            conv_C2C(d_input
                , d_kernel
                , N
                , dx
                , stream
                , d_output
                );
            break;
    }

    checkDeviceMemory();
    timer.stop();
    timer.printElapsedTime();

    cudaStreamSynchronize(stream);

    std::vector<float> h_output(N);
    cudaMemcpyAsync(h_output.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost, stream); 

    for (auto i = 0u; i < N; ++i) {
        std::cout << i << " : " << h_output[i] << " " << h_output_theory[i] << std::endl;
        if (abs(h_output[i] - h_output_theory[i]) > 0.01)
        {
            std::cout << "large numerical differences! test failed" << std::endl;
        }
    }
    std::cout << "accuracy test: success" << std::endl;

    std::cout << std::endl;
    
//    // plot
//    plt::plot(h_k, h_fk_real);
//    plt::show();

    // clean up
    cudaFreeAsync(d_output, stream);    

    return 0;
}

