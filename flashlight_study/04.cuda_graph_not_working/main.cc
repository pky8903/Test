// main.cc
#include <flashlight/fl/flashlight.h>
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <flashlight/fl/runtime/CUDAStream.h>

using namespace fl;

static inline void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f;
    check(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

int main(int argc, char** argv) {
    fl::init();
    fl::setSeed(1);

    fl::setDevice(0);
    auto native_id = fl::getDevice();
    cudaSetDevice(native_id);

    std::cout << "native_id : " << native_id << std::endl;

    int N = (argc > 1) ? std::atoi(argv[1]) : 4;
    int C = (argc > 2) ? std::atoi(argv[2]) : 64;
    int H = (argc > 3) ? std::atoi(argv[3]) : 1024;
    int W = (argc > 4) ? std::atoi(argv[4]) : 1024;
    int iters = (argc > 5) ? std::atoi(argv[5]) : 20;

    std::cout << "Shape: N=" << N << " C=" << C << " H=" << H << " W=" << W
        << " iters=" << iters << std::endl;

    cudaEvent_t evStart, evStop;    check(cudaEventCreate(&evStart));
    cudaEvent_t evStart2, evStop2;  check(cudaEventCreate(&evStart2)); 

    auto x  = Variable(fl::randn({W, H, C, N}), false);
    auto s1 = Variable(fl::full({1, 1, C, 1}, 1.1f), false);
    auto b1 = Variable(fl::full({1, 1, C, 1}, 0.1f), false);

    auto x_tensor = x.tensor();
    const auto& fl_stream = x_tensor.stream();
    auto stream = fl_stream.impl<fl::CUDAStream>().handle();

    auto conv1 = std::make_shared<fl::Conv2D>(C, C, 3, 3, 1, 1, 1, 1, 1, 1, 1, true);
    auto conv2 = std::make_shared<fl::Conv2D>(C, C, 1, 1, 1, 1, 0, 0, 1, 1, 1, true);

    conv1->eval();
    conv2->eval();

    auto heavy_opts = [&](const Variable& in) {
        auto y  = fl::relu(in * s1 + b1); 
        auto x2 = y * y;
        auto x3 = x2 * y;
        auto gelu = 0.5f * y * (1.0f + fl::tanh(0.79788456f * (y + 0.044715 * x3)));
        auto z = gelu * gelu;

        auto c1 = conv1->forward(z);
        auto c2 = conv2->forward(c1);

        auto out = fl::sin(c2) + fl::cos(c2 * 0.5f);
        auto sum = fl::sum(out, {0, 1, 2, 3}, true);
        return sum;
    };

    // warm up
    for (int i = 0; i < 3; ++i) {
        auto s = heavy_opts(x);
        fl::eval(s.tensor());
    } 
    check(cudaDeviceSynchronize());
    fl::eval(x.tensor());

//    // without graph
//    fl::eval(x.tensor());
//    Variable sum_eager;
//    check(cudaEventRecord(evStart2, stream));
//    for (int i = 0; i < iters; ++i) {
//        nvtxRangePush("cudagraph");
//        sum_eager = heavy_opts(x);
//        fl::eval(sum_eager.tensor());
////        check(cudaDeviceSynchronize()); 
//        nvtxRangePop();
//    }
//
//    check(cudaEventRecord(evStop2, stream));
//    check(cudaStreamSynchronize(stream));
//
//    float eager_ms = elapsed_ms(evStart2, evStop2);
//    float sum_val_eager = sum_eager.tensor().scalar<float>();

    // cuda graph capture
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    nvtxRangePush("StreamCapture");

    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    Variable sum_graph; // cuda malloc async

    {
        sum_graph = heavy_opts(x);       
        fl::eval(sum_graph.tensor());
    }

    check(cudaStreamEndCapture(stream, &graph));
    check(cudaStreamSynchronize(stream)); 
    nvtxRangePop();
    std::cout << "capture end!" << std::endl;

    check(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    std::cout << "graph instantiate!" << std::endl; 

    // cuda graph
    check(cudaEventRecord(evStart, stream));
    for (int i = 0; i < iters; ++i) {
        nvtxRangePush("cudagraph");
        check(cudaGraphLaunch(graphExec, stream));
        nvtxRangePop();
    }
    
    check(cudaEventRecord(evStop, stream));
    check(cudaStreamSynchronize(stream)); 
    float graph_ms = elapsed_ms(evStart, evStop);

//    float sum_val_graph = sum_graph.tensor().scalar<float>();

//    std::cout << "sum (graph) = " << sum_val_graph << "\n";
//    std::cout << "sum (eager) = " << sum_val_eager << "\n";
    std::cout << "Graph time  : " << graph_ms  << " ms for " << iters << " iters ("
              << (graph_ms/iters) << " ms/iter)\n";
//    std::cout << "Eager time  : " << eager_ms  << " ms for " << iters << " iters ("
//              << (eager_ms/iters) << " ms/iter)\n";
//    std::cout << "Speedup     : " << (eager_ms / graph_ms) << " x\n";
  
    // cleanup
    check(cudaGraphExecDestroy(graphExec));
    check(cudaGraphDestroy(graph));
    check(cudaEventDestroy(evStart)); check(cudaEventDestroy(evStop));
    check(cudaEventDestroy(evStart2)); check(cudaEventDestroy(evStop2));

    return 0;
} 
