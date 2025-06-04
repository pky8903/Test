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

inline void checkCufftError(cufftResult result, const char* msg) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "[cuFFT Error] " << msg << ": ";
        switch (result) {
            case CUFFT_INVALID_PLAN:
                std::cerr << "Invalid plan handle." << std::endl; break;
            case CUFFT_ALLOC_FAILED:
                std::cerr << "Memory allocation failed." << std::endl; break;
            case CUFFT_INVALID_TYPE:
                std::cerr << "Invalid type." << std::endl; break;
            case CUFFT_INVALID_VALUE:
                std::cerr << "Invalid value." << std::endl; break;
            case CUFFT_INTERNAL_ERROR:
                std::cerr << "Internal cuFFT error." << std::endl; break;
            case CUFFT_EXEC_FAILED:
                std::cerr << "FFT execution failed." << std::endl; break;
            case CUFFT_SETUP_FAILED:
                std::cerr << "cuFFT library setup failed." << std::endl; break;
            case CUFFT_INVALID_SIZE:
                std::cerr << "Invalid transform size." << std::endl; break;
            case CUFFT_UNALIGNED_DATA:
                std::cerr << "Unaligned data error." << std::endl; break;
            default:
                std::cerr << "Unknown error code: " << result << std::endl; break;
        }
        std::exit(EXIT_FAILURE);
    }
}

void generateGaussian(
    std::vector<float>& data
    , int width
    , int height
    , float x0 // shift
    , float y0 // shift
    , float dx
    , float sigma
    , float coeff = 1.f
    ) 
{
    for (auto j = 0; j < height; ++j) {
        for (auto i = 0; i < width; ++i) {
            auto ind = width * j + i;
            auto x = (i - width / 2) * dx;
            auto y = (j - height / 2) * dx;
            data[ind] = coeff * expf(- (x - x0) * (y - y0) / (2.f * sigma * sigma));
        }
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
__global__ void fftshift_2D(const T* __restrict__ input
    , const int width, const int height
    , T* __restrict__ output
    ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width or y >= height) return;

    int shifted_x = (x + width / 2) % width;
    int shifted_y = (y + height / 2) % height;

    int input_idx = x * height + y;
    int output_idx = shifted_x * height + shifted_y;

    output[output_idx] = input[input_idx];
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

// out-of-place
template<typename T>
void api_fftshift_2D(T* input, const int width, const int height, cudaStream_t stream, T* output) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
    fftshift_2D<<<grid, block, 0, stream>>>(input, width, height, output);
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
void api_fftC2C_2D(cufftComplex* in, cufftComplex*out, const int width, const int height, int direction, cudaStream_t stream)
{
    GpuTimer timer("api_fftC2C_2D");
    timer.start();

    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CUFFT_C2C);
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

void api_fftR2C_2D(float* in, cufftComplex* out, const int width, const int height, int direction, cudaStream_t stream)
{
    GpuTimer timer("api_fftR2C_2D");
    timer.start();

    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CUFFT_R2C);
    cufftSetStream(plan, stream);
    cufftExecR2C(plan, in, out);
    cufftDestroy(plan);

    cudaStreamSynchronize(stream);
    timer.stop();
    timer.printElapsedTime();
}

void api_fftC2R_2D(cufftComplex* in, float* out, const int width, const int height, int direction, cudaStream_t stream)
{
    GpuTimer timer("api_fftC2R_2D");
    timer.start();

    cufftHandle plan;
    checkCufftError(cufftPlan2d(&plan, height, width, CUFFT_C2R), "fft_c2r_2D");
    cufftSetStream(plan, stream);
    cufftExecC2R(plan, in, out);
    cufftDestroy(plan);

    cudaStreamSynchronize(stream);
    timer.stop();
    timer.printElapsedTime();
}

void conv_C2C(float* d_input
    , float* d_kernel
    , const int N 
    , const float dx
    , cudaStream_t stream
    , float* d_output
    )
{
    auto size = N * N;

    float* d_shifted_input = nullptr;
    cudaMallocAsync(&d_shifted_input, sizeof(float) * size, stream);

    float* d_shifted_kernel = nullptr;
    cudaMallocAsync(&d_shifted_kernel, sizeof(float) * size, stream);

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
    api_fftshift_2D<float>(d_input, N, N, stream, d_shifted_input);
    api_fftshift_2D<float>(d_kernel, N, N, stream, d_shifted_input);

    // make_complex
    api_makeComplex(d_shifted_input, d_cplx_input, size, stream);
    api_makeComplex(d_shifted_kernel, d_cplx_kernel, size, stream);
    cudaFreeAsync(d_shifted_input, stream);
    cudaFreeAsync(d_shifted_kernel, stream);

    // fft
    api_fftC2C_2D(d_cplx_input, d_ffted_input, N, N, CUFFT_FORWARD, stream);
    api_fftC2C_2D(d_cplx_kernel, d_ffted_kernel, N, N, CUFFT_FORWARD, stream);
    cudaFreeAsync(d_cplx_input, stream);
    cudaFreeAsync(d_cplx_kernel, stream);

    // point-wise multiplication
    api_mul<cufftComplex>(d_ffted_input, d_ffted_kernel, size, stream); 
    auto& d_ffted_output = d_ffted_input;
    cudaFreeAsync(d_ffted_kernel, stream);
    
    // ifft
    api_fftC2C_2D(d_ffted_output, d_cplx_output, N, N, CUFFT_INVERSE, stream);

    // to real
    float* d_shifted_output = nullptr;
    cudaMallocAsync(&d_shifted_output, sizeof(float) * size, stream);

    api_makeReal(d_cplx_output, d_shifted_output, size, stream);
    cudaFreeAsync(d_cplx_output, stream);

    // ifftshift
    api_fftshift_2D<float>(d_shifted_output, N, N, stream, d_output);
    cudaFreeAsync(d_shifted_output, stream);

    // scale
    api_scale<float>(d_output, 1.f/(static_cast<float>(size)), size, stream);
}

void conv_R2C(float* d_input
    , float* d_kernel
    , const int N
    , const float dx
    , cudaStream_t stream
    , float* d_output
    )
{
    auto size = N * N;

    cufftComplex* d_ffted_input = nullptr;
    cudaMallocAsync(&d_ffted_input, sizeof(cufftComplex) * N * (N / 2 + 1), stream);

    cufftComplex* d_ffted_kernel = nullptr;
    cudaMallocAsync(&d_ffted_kernel, sizeof(cufftComplex) * N * (N / 2 + 1), stream);

    float* d_shifted_input = nullptr;
    cudaMallocAsync(&d_shifted_input, sizeof(float) * size, stream);
    
    float* d_shifted_kernel = nullptr;
    cudaMallocAsync(&d_shifted_kernel, sizeof(float) * size, stream);
    
    // fftshift
    api_fftshift_2D<float>(d_input, N, N, stream, d_shifted_input);
    api_fftshift_2D<float>(d_kernel, N, N, stream, d_shifted_kernel);

    // fft
    api_fftR2C_2D(d_shifted_input, d_ffted_input, N, N, CUFFT_FORWARD, stream);
    api_fftR2C_2D(d_shifted_kernel, d_ffted_kernel, N, N, CUFFT_FORWARD, stream);
    cudaFreeAsync(d_shifted_input, stream);
    cudaFreeAsync(d_shifted_kernel, stream);

    // point-wise multiplicataion
    api_mul<cufftComplex>(d_ffted_input, d_ffted_kernel, size, stream);
    auto& d_ffted_output = d_ffted_input;

    // ifft
    float* d_shifted_output = nullptr;
    cudaMallocAsync(&d_shifted_output, sizeof(float) * size, stream);
    api_fftC2R_2D(d_ffted_output, d_shifted_output, N, N, CUFFT_INVERSE, stream);
    
    // fftshift
    api_fftshift_2D<float>(d_shifted_output, N, N, stream, d_output);
    cudaFreeAsync(d_shifted_output, stream);
    
    // scale
    api_scale<float>(d_output, 1.f/(static_cast<float>(size)), size, stream);

    // clean-up
    cudaFreeAsync(d_ffted_input, stream);
    cudaFreeAsync(d_ffted_kernel, stream);
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    
    // data preparation 
    const int N = parse_option(argc, argv, "--width");
    const int mode = parse_option(argc, argv, "--mode");
    const float dx = 8.f;
    const float sigma = 100.f;
    const float sigma_k = 50.f;
    const float width = dx * N;
    const float dk = 2.f * pi / (dx * N);
    const float x0 = 14.f * dx;
    const float y0 = 14.f * dx;
    
    auto normalize_factor = [](const float sigma) {
        return 1.f/(2.f * pi * sigma * sigma);
    };

    auto h_input = std::vector<float>(N * N);
    generateGaussian(h_input, N, N, x0, y0, dx, sigma, normalize_factor(sigma));

    auto h_kernel = std::vector<float>(N * N);
    generateGaussian(h_kernel, N, N, 0.f, 0.f, dx, sigma_k, normalize_factor(sigma_k));

    auto sigma_o = sqrt(sigma * sigma + sigma_k * sigma_k);
    auto h_output_theory = std::vector<float>(N * N);
    generateGaussian(h_output_theory, N, N, x0, y0, dx, sigma_o, normalize_factor(sigma_o));
    
    // cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // gpu memory allocation
    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    cudaMallocAsync(&d_input, sizeof(float) * N * N, stream);
    cudaMallocAsync(&d_kernel, sizeof(float) * N * N, stream);
    cudaMallocAsync(&d_output, sizeof(float) * N * N, stream);

    cudaMemcpyAsync(d_input, h_input.data(), sizeof(float) * N * N, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kernel, h_kernel.data(), sizeof(float) * N * N, cudaMemcpyHostToDevice, stream);

    // for GPU warming up
    std::cout << "GPU warming up - start" << std::endl;
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

    std::vector<float> h_output(N * N);
    cudaMemcpyAsync(h_output.data(), d_output, sizeof(float) * N * N, cudaMemcpyDeviceToHost, stream); 

    for (auto i = 0u; i < N * N; ++i) {
        if (abs(h_output[i] - h_output_theory[i]) > 0.1)
        {
            std::cerr << "large numerical differences! test failed" << std::endl;
            return 0;
        }
    }
    std::cout << "accuracy test: success" << std::endl;

    std::cout << std::endl;

    // clean up
    cudaFreeAsync(d_input, stream);
    cudaFreeAsync(d_kernel, stream);
    cudaFreeAsync(d_output, stream);    

    return 0;
}

