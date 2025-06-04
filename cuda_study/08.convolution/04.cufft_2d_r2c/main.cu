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

void plotColumnMajorFloatImage(
    const std::vector<float>& data,
    int width,
    int height
) {
    // 1) reshape into vector<vector<double>> [row][col]
    std::vector<std::vector<double>> img;
    img.assign(height, std::vector<double>(width));
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            // index into column-major flat array
            int idx = col * height + row;
            img[row][col] = static_cast<double>(data[idx]);
        }
    }

//    // 2) plot
//    plt::figure_size(width*4, height*4);        // scale up so you can see it
//    plt::imshow(img, {{"interpolation", "none"}});
//    plt::colorbar();
//    plt::show();
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
    std::cout << "generateGaussian with sigma: " << sigma << " and the peak value of " << coeff << std::endl;
    for (auto j = 0; j < height; ++j) {
        for (auto i = 0; i < width; ++i) {
            auto ind = width * j + i;
            auto x = (i - width / 2) * dx;
            auto y = (j - height / 2) * dx;
            data[ind] = coeff * expf(- ((x - x0) * (x - x0) + (y - y0) * (y - y0))/ (2.f * sigma * sigma));
        }
    }
}

void generateAnswer(
    std::vector<cufftComplex>& data 
    , int width
    , int height
    , float x0
    , float y0
    , float dk 
    , float sigma
    , float coeff
    )
{
    for (auto j = 0; j < height; ++j) { 
       for (auto i = 0; i < width; ++i) {
            auto ind = j * width + i;
            auto kx = (i - width / 2) * dk;
            auto ky = (j - height / 2) * dk;
            auto envelope = coeff * expf(- 0.5f * ( kx * kx + ky * ky) * sigma * sigma);
            data[ind].x = envelope * cosf(-kx*x0-ky*y0);
       	    data[ind].y = envelope * sinf(-kx*x0-ky*y0);
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

template<typename T>
void printHostImage(const std::vector<T>& in, const int width, const int height, const int x1, const int y1, const int x2, const int y2, const std::string& tag){}

template<>
void printHostImage<float>(const std::vector<float>& data,
                       const int width,
                       const int height,
                       const int x1,
                       const int y1,
                       const int x2,
                       const int y2,
                       const std::string& tag) {

    std::cout << "[printImage] " << tag << ": "
              << "x1=" << x1 << ", y1=" << y1
              << ", x2=" << x2 << ", y2=" << y2
              << ", width=" << width << ", height=" << height << std::endl;

    if (x1 < 0 || x2 > width || y1 < 0 || y2 > height) {
        std::cerr << "[printImage] Invalid range requested" << std::endl;
        return;
    }

    for (int j = y1; j < y2; ++j) {
        for (int i = x1; i < x2; ++i) {
            int ind = j * width + i;
            if (ind < 0 || ind >= width * height) {
                std::cerr << "[printImage] Invalid index: " << ind << std::endl;
                continue;
            }
            std::cout << "[" << i << "," << j << "] = "
                      << data[ind] << std::endl;
            std::cout.flush(); 
        }
    }

    std::cout << "[printImage] done." << std::endl;
}

template<>
void printHostImage<cufftComplex>(const std::vector<cufftComplex>& data,
                              const int width,
                              const int height,
                              const int x1,
                              const int y1,
                              const int x2,
                              const int y2,
                              const std::string& tag) {
    std::cout << "[printImage] " << tag << ": "
              << "x1=" << x1 << ", y1=" << y1
              << ", x2=" << x2 << ", y2=" << y2
              << ", width=" << width << ", height=" << height << std::endl;

    if (x1 < 0 || x2 > width || y1 < 0 || y2 > height) {
        std::cerr << "[printImage] Invalid range requested" << std::endl;
        return;
    }

    for (int j = y1; j < y2; ++j) {
        for (int i = x1; i < x2; ++i) {
            int ind = j * width + i;
            if (ind < 0 || ind >= width * height) {
                std::cerr << "[printImage] Invalid index: " << ind << std::endl;
                continue;
            }
            std::cout << "[" << i << "," << j << "] = "
                      << data[ind].x << " + " << data[ind].y << "i" << std::endl;
            std::cout.flush();
        }
    }

    std::cout << "[printImage] done." << std::endl;
}

template<typename T>
void printImage(T* in, const int width, const int height, const int x1, const int y1, const int x2, const int y2, cudaStream_t stream, const std::string& tag){}

template<>
void printImage<float>(float* data,
                       const int width,
                       const int height,
                       const int x1,
                       const int y1,
                       const int x2,
                       const int y2,
                       cudaStream_t stream,
                       const std::string& tag) {
    std::vector<float> h_data(width * height);

    cudaError_t err = cudaMemcpyAsync(
        h_data.data(), data,
        sizeof(float) * width * height,
        cudaMemcpyDeviceToHost, stream);

    if (err != cudaSuccess) {
        std::cerr << "[printImage] cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);

    printHostImage<float>(h_data, width, height, x1, y1, x2, y2, tag);
    plotColumnMajorFloatImage(h_data, width, height);
}

template<>
void printImage<cufftComplex>(cufftComplex* data,
                              const int width,
                              const int height,
                              const int x1,
                              const int y1,
                              const int x2,
                              const int y2,
                              cudaStream_t stream,
                              const std::string& tag) {
    std::vector<cufftComplex> h_data(width * height);

    cudaError_t err = cudaMemcpyAsync(
        h_data.data(), data,
        sizeof(cufftComplex) * width * height,
        cudaMemcpyDeviceToHost, stream);

    if (err != cudaSuccess) {
        std::cerr << "[printImage] cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);
    
    printHostImage<cufftComplex>(h_data, width, height, x1, y1, x2, y2, tag);
}

constexpr double pi = 3.141592655358979323846;

int main(int argc, char* argv[])
{
    // data preparation 
    const int N = parse_option(argc, argv, "--width");
    const float dx = 8.f;
    const float sigma = 9.f;
    const float width = dx * N;
    const float dk = 2.f * pi / (dx * N);
    const float x0 = 0.f * dx;
    const float y0 = 0.f * dx;
    
    auto normalize_factor = [](const float sigma) {
        return 1.f/(2.f * pi * sigma * sigma);
    };

    auto h_input = std::vector<float>(N * N);
    generateGaussian(h_input, N, N, x0, y0, dx, sigma, normalize_factor(sigma));

    auto h_output_theory = std::vector<cufftComplex>(N * N);

    generateAnswer(
        h_output_theory
	    , N, N
	    , x0, y0
	    , dk
	    , sigma
	    , 1.f
	    );
    
    // cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // gpu memory allocation
    float* d_input = nullptr;
    float* d_shifted_input = nullptr;
    cufftComplex* d_cplx_input = nullptr;
    cufftComplex* d_output = nullptr;
    cufftComplex* d_shifted_output = nullptr;
    cufftComplex* d_output_r2c = nullptr;

    cudaMallocAsync(&d_input, sizeof(float) * N * N, stream);
    cudaMallocAsync(&d_shifted_input, sizeof(float) * N * N, stream);
    cudaMallocAsync(&d_cplx_input, sizeof(cufftComplex) * N * N, stream);
    cudaMallocAsync(&d_shifted_output, sizeof(cufftComplex) * N * N, stream);
    cudaMallocAsync(&d_output, sizeof(cufftComplex) * N * N, stream);
    cudaMallocAsync(&d_output_r2c, sizeof(cufftComplex) * N * (N/2 + 1), stream);
    cudaMemcpyAsync(d_input, h_input.data(), sizeof(float) * N * N, cudaMemcpyHostToDevice, stream);

    GpuTimer timer("main algorithm");
    timer.start();

    // fftshift
    api_fftshift_2D<float>(d_input, N, N, stream, d_shifted_input);
    
    // make complex
    api_makeComplex(d_shifted_input, d_cplx_input, N * N, stream);
    cudaFreeAsync(d_input, stream);    

    // fft
    api_fftC2C_2D(d_cplx_input, d_shifted_output, N, N, CUFFT_FORWARD, stream);
    cudaFreeAsync(d_cplx_input, stream);    

    // fft r2c test
    api_fftR2C_2D(d_shifted_input, d_output_r2c, N, N, CUFFT_FORWARD, stream);

    // fftshift
    api_fftshift_2D<cufftComplex>(d_shifted_output, N, N, stream, d_output); 
    cudaFreeAsync(d_shifted_output, stream);

    // scale
    api_scale<cufftComplex>(d_output, make_cuComplex(dx*dx, 0.f), N * N, stream);
    api_scale<cufftComplex>(d_output_r2c, make_cuComplex(dx*dx, 0.f), N * (N / 2 + 1), stream);

    checkDeviceMemory();
    timer.stop();
    timer.printElapsedTime();

    cudaStreamSynchronize(stream);

    // copy to host
    std::vector<cufftComplex> h_fk(N * N);
    cudaMemcpyAsync(h_fk.data(), d_output, sizeof(cufftComplex) * N * N, cudaMemcpyDeviceToHost, stream); 

    std::vector<cufftComplex> h_fk_r2c(N * (N/2 + 1) );
    cudaMemcpyAsync(h_fk_r2c.data(), d_output_r2c, sizeof(cufftComplex) * (N / 2 + 1), cudaMemcpyDeviceToHost, stream); 

    cudaStreamSynchronize(stream);

    auto x1 = N/2 + 0;
    auto y1 = N/2 + 0;
    auto x2 = N/2 + 3;
    auto y2 = N/2 + 3;
    
    printHostImage<cufftComplex>(h_output_theory, N, N, x1, y1, x2, y2, "h_fk_theory");
    printImage<cufftComplex>(d_output, N, N, x1, y1, x2, y2, stream, "h_fk");
    printImage<cufftComplex>(d_output_r2c, N / 2 + 1, N, 0, 0, 3, 3, stream, "h_fk_r2c");

    // clean up
    cudaFreeAsync(d_output, stream);    
    cudaFreeAsync(d_output_r2c, stream);    

    return 0;
}

