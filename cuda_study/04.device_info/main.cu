#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>


#define _1MB (1024*1024)

int main(void) {
    int ngpus;
    cudaGetDeviceCount(&ngpus);

    for (int i = 0; i < ngpus; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
    
        printf("\tDevice %d: %s\n", i, devProp.name);
        printf("\tCompute capability: %d.%d\n", devProp.major, devProp.minor);
        printf("\tThe number of streaming multiprocessors: %d\n", devProp.multiProcessorCount);
        printf("\tThe number of CUDA cores: %d\n", _ConvertSMVer2Cores(devProp.major, devProp.minor) * devProp.multiProcessorCount);
        printf("\tGlobal memory size: %.2f MB\n", (float)devProp.totalGlobalMem / _1MB);
    }
    return 0;
}
