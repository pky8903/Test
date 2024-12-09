#include <iostream>
#include <cufft.h>
#include <complex> 
#include <random>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <cuComplex.h>
#include <nvtx3/nvToolsExt.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"

#define PRINT_AI 0

#define cudaCheckError() {                                  \
    cudaError_t e = cudaGetLastError();                     \
    if (e != cudaSuccess) {                                 \
        printf("CUDA error %s %d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(e));     \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}                                                           \

void checkCufft(cufftResult result, const char* msg) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error (" << msg << "): ";
        switch (result) {
            case CUFFT_INVALID_PLAN: std::cerr << "Invalid plan"; break;
            case CUFFT_ALLOC_FAILED: std::cerr << "Allocation failed"; break;
            case CUFFT_INVALID_VALUE: std::cerr << "Invalid value"; break;
            case CUFFT_INTERNAL_ERROR: std::cerr << "Internal error"; break;
            case CUFFT_EXEC_FAILED: std::cerr << "Execution failed"; break;
            default: std::cerr << "Unknown error"; break;
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory (free/total) = %zu/%zu bytes\n", free, total);
}

void generateRandomImageColumnMajor(
    int width
    , int height
    , int batch
    , std::vector<cuFloatComplex >& image
    , const int seed
    )
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    auto size = width * height * batch;
    image.resize(size);
    for (int nb = 0; nb < batch; ++nb) {
        for (int col = 0; col < height; ++col) {
            for (int row = 0; row < width; ++row) {
                int idx = nb * width * height + col * width + row;
                float real_val = dis(gen);
                float imag_val = dis(gen);
                image[idx] = make_cuFloatComplex(real_val, imag_val);
            }
        }
    }
}

void generateRandomKernelsColumnMajor(
    int width
    , int height
    , int batch
    , int n_terms
    , std::vector<std::vector<cuFloatComplex > >& kernels
    , std::vector<float>& eigen_values
    )
{
    kernels.resize(n_terms);
    eigen_values.resize(n_terms);

    for (auto i = 0; i < n_terms; ++i) {
        generateRandomImageColumnMajor(
            width
            , height
            , batch
            , kernels[i]
            , i
            );
        eigen_values[i] = std::exp(-10*i/static_cast<float>(n_terms)); 
        // The eigenvalue distribution should vary depending on the batch ID
        // , but for now, I assume it to be the same.
    }
}

struct vectorMul {
    __host__ __device__
    cuFloatComplex operator()(const cuFloatComplex in1
        , const cuFloatComplex in2
        ) const
    {
        return cuCmulf(in1, in2);
    }
};

struct aiAccumulator {
    const float weight;
    explicit aiAccumulator(float _weight) : weight(_weight) {}

    __host__ __device__
    float operator()(thrust::tuple<float, cuFloatComplex> input) {
        auto ai = thrust::get<0>(input);    
        auto conv = thrust::get<1>(input); 
        return ai + weight * (conv.x * conv.x + conv.y * conv.y);
    }
};

struct complexDiffSquared {
    __host__ __device__
    float operator()(const cuFloatComplex& a, const cuFloatComplex& b) const {
        float real_diff = a.x - b.x;
        float imag_diff = a.y - b.y;
        return real_diff * real_diff + imag_diff * imag_diff;
    }
};

struct scaling {
    const float scale;
    explicit scaling(float _scale) : scale(_scale) {}
    
    __host__ __device__
    cuFloatComplex operator()(const cuFloatComplex& val) const {
        float real = val.x * scale;
        float imag = val.y * scale;
        return make_cuFloatComplex(real, imag);
    }
};

float calculate_rms(cuFloatComplex* arr1, cuFloatComplex* arr2, const int size) {
    thrust::device_ptr<cuFloatComplex> t_arr1(arr1);
    thrust::device_ptr<cuFloatComplex> t_arr2(arr2);
    thrust::device_vector<float> squared_diff(size);
    thrust::transform(t_arr1, t_arr1 + size, t_arr2, squared_diff.begin(), complexDiffSquared());
    float sum = thrust::reduce(squared_diff.begin(), squared_diff.end(), 0.0f, thrust::plus<float>());
    return std::sqrt(sum);
}

DS_timer timer(4);
void setTimer();

int main(void)
{
    std::cout << "Start compute_ai" << std::endl;
    checkDeviceMemory();
    setTimer();
    
    // Data size
    const int width = 1024 * 4;     // 256 for calibration, 1024, 2048, 4096 for simulation     
    const int height = 1024 * 4;    // 256 for calibration, 1024, 2048, 4096 for simulation
    const int batch = 5;
    const int size = width * height * batch;

    // Kernel info
    const int n_terms = 64;     // typically 64, maximally 128

    std::cout << "Size: width = " << width << ", height = " << height << std::endl;
    std::cout << "TCC terms: " << n_terms << std::endl;
    
    // create host image data
    auto mi = std::vector<cuFloatComplex >{};
    generateRandomImageColumnMajor(
        width
        , height
        , batch   
        , mi
        , 1         /* deterministic random seed */
        );

    // create host kernel data
    auto kernels = std::vector<std::vector<cuFloatComplex > >{};
    auto eigen_values = std::vector<float>{};
    generateRandomKernelsColumnMajor(
        width
        , height
        , batch
        , n_terms
        , kernels
        , eigen_values
        );
    
    timer.onTimer(0);
    timer.onTimer(2);

    // Host-to-device copy 
    float* d_ai = nullptr;
    cuFloatComplex* d_mi;
    cuFloatComplex* d_mi_fft;
    cuFloatComplex* d_buffer;
    std::vector<cuFloatComplex*> d_kernels(n_terms);

    cudaMalloc((void**)&d_ai, size*sizeof(float));
    cudaMemset(d_ai, 0.f, size*sizeof(float));

    cudaMalloc(
        &d_mi
        , mi.size() * sizeof(cuFloatComplex)
        );
    cudaMemcpy(
        d_mi
        , mi.data()
        , mi.size() * sizeof(cuFloatComplex)
        , cudaMemcpyHostToDevice
        );

    cudaMalloc(
        &d_mi_fft
        , mi.size() * sizeof(cuFloatComplex)
        );
    cudaMalloc(
        &d_buffer
        , mi.size() * sizeof(cuFloatComplex)
        );

    for (auto i = 0u; i < d_kernels.size(); ++i) {
        cudaMalloc(
            &d_kernels[i]
            , kernels[i].size() * sizeof(cuFloatComplex)
            );
        cudaMemcpy(
            d_kernels[i]
            , kernels[i].data()
            , kernels[i].size() * sizeof(cuFloatComplex)
            , cudaMemcpyHostToDevice
            );
    }
    timer.offTimer(2);

    timer.onTimer(1);
    nvtxRangePush("Main algorithm");
    // main algorithm (to be accelerated by utilizing TensorCore)
    // : AI = Sum of ( w_i * abs(IFFT{ FFT{MI} * K_i }) * abs(IFFT{ FFT{MI} * K_i }) )

    std::cout << "Start computing the main algorithm" << std::endl;
    // 1. Fourier transform of MI
    nvtxRangePush("CUFFT MI");
    int size_array[2] = { width, height }; 
    cufftHandle plan;
    checkCufft(
        cufftPlanMany(&plan
            , 2 /* fft dimension */
            , size_array
            , NULL, 1, width * height
            , NULL, 1, width * height
            , CUFFT_C2C, batch
            )
        , "cufftPlanMany"
        );
    checkCufft(cufftExecC2C(plan, d_mi, d_mi_fft, CUFFT_FORWARD), "cufftExecC2C (Forward)");
    nvtxRangePop();
    
    // 2. TCC convolution
    for (auto i = 0; i < n_terms; ++i) {
        // FFT convolution
        nvtxRangePush("Fourier domain multiplication");
        thrust::device_ptr<cuFloatComplex> t_kernel(d_kernels[i]);
        thrust::device_ptr<cuFloatComplex> t_mi_fft(d_mi_fft);
        thrust::device_ptr<cuFloatComplex> t_buffer(d_buffer);
        thrust::transform(thrust::device
            , t_mi_fft
            , t_mi_fft + size
            , t_kernel
            , t_buffer
            , vectorMul()
        );       
        nvtxRangePop();

        nvtxRangePush("CUFFT IFFT");
        checkCufft(cufftExecC2C(plan, d_buffer, d_buffer, CUFFT_INVERSE), "cufftExecC2C (Inverse)");
        nvtxRangePop();

        // accumulation 
        // ai += weight_i * abs(d_buffer)*abs(d_buffer)
                
        nvtxRangePush("Accumulation");
        thrust::device_ptr<float> t_ai(d_ai);
        auto begin = thrust::make_zip_iterator(
            thrust::make_tuple(t_ai, t_buffer)
            );
        auto end = thrust::make_zip_iterator(
            thrust::make_tuple(t_ai + size, t_buffer + size)
            );
        thrust::transform(thrust::device
            , begin
            , end
            , t_ai
            , aiAccumulator(eigen_values[i]/size)
            );
        nvtxRangePop();
    }

    timer.offTimer(1);
    nvtxRangePop();
    std::cout << "main algorithm finished!" << std::endl;

    // Device-to-host copy
    timer.onTimer(3);
    thrust::device_ptr<float> t_ai(d_ai);
    thrust::host_vector<float> h_ai(size); 
    thrust::copy(t_ai, t_ai + size, h_ai.begin());
    std::vector<float> ai(h_ai.begin(), h_ai.end()); 
    timer.offTimer(3);

    timer.offTimer(0);

#if PRINT_AI
    // print results
    std::cout << "computed ai" << std::endl;
    for (auto nb = 0; nb < batch; ++nb) {
        std::cout << "batch id: " << nb << std::endl;
        for (auto nx = 0; nx < width; ++nx) {
            for (auto ny = 0; ny < height; ++ny) {
                auto ind = nb * width * height + ny * width + nx;
                std::cout << ai[ind] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    timer.printTimer(1);

    // free memory 
    cufftDestroy(plan);
    cudaFree(d_ai);
    cudaFree(d_mi);
    cudaFree(d_mi_fft);
    cudaFree(d_buffer);
    for (auto i = 0; i < n_terms; ++i) { 
        cudaFree(d_kernels[i]);
    }
 
    cudaDeviceSynchronize();
    checkDeviceMemory();
    std::cout << "Succesfully completed" << std::endl;
    return 0;
}

void setTimer()
{
    timer.setTimerName(0, (char*)"Total COMPUTE_AI algorithm");    
    timer.setTimerName(1, (char*)" - main algorithm");
    timer.setTimerName(2, (char*)" - [Data transfer] host->device");
    timer.setTimerName(3, (char*)" - [Data transfer] device->host");
}


