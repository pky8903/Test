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
    // fl init
    fl::init();
    fl::setSeed(1);
    fl::setDevice(0);

    auto native_id = fl::getDevice();
    cudaSetDevice(native_id);

    std::cout << "native_id : " << native_id << std::endl;

    // parse input arg
    int N = (argc > 1) ? std::atoi(argv[1]) : 4;
    int C = (argc > 2) ? std::atoi(argv[2]) : 64;
    int H = (argc > 3) ? std::atoi(argv[3]) : 1024;
    int W = (argc > 4) ? std::atoi(argv[4]) : 1024;
    int iters = (argc > 5) ? std::atoi(argv[5]) : 20;

    std::cout << "Shape: N=" << N << " C=" << C << " H=" << H << " W=" << W
        << " iters=" << iters << std::endl;

    // prepare input var and forward modules
    auto x  = Variable(fl::randn({W, H, C, N}), /*calc_grad=*/ false);
    auto s1 = Variable(fl::full({1, 1, C, 1}, 1.0f), false);
    auto b1 = Variable(fl::full({1, 1, C, 1}, 0.1f), false);

    auto conv1 = std::make_shared<fl::Conv2D>(C, C, 3, 3, 1, 1, 1, 1, 1, 1, 1, true);
    auto conv2 = std::make_shared<fl::Conv2D>(C, C, 1, 1, 1, 1, 0, 0, 1, 1, 1, true);    

    conv1->eval(); // forward-only
    conv2->eval(); // forward-only

    auto heavy_ops = [&](const Variable& in) {
        auto y = fl::relu(in * s1 + b1);    
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

    // warm up and test
    fl::eval(x.tensor());
    
    for (auto i = 0; i < 3; ++i) {
        nvtxRangePush("natural heavy_ops");
        auto s = heavy_ops(x);
        fl::sync();
        nvtxRangePop();
    }

    // runtime measurement without graph
    fl::eval(x.tensor());

    Variable sum_eager;
    auto start = fl::Timer::start();
    for (auto i = 0; i < iters; ++i) {
        nvtxRangePush("sum eager");
        sum_eager = heavy_ops(x);
        fl::sync();
        nvtxRangePop();
    }
    auto eager_ms = fl::Timer::stop(start);
    auto sum_val_eager = sum_eager.tensor().scalar<float>();

    // cuda graph capture
    auto x_tensor = x.tensor();
    const auto& fl_stream = x_tensor.stream();
    auto stream = fl_stream.impl<fl::CUDAStream>().handle();

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    nvtxRangePush("Graph capture");
    Variable sum_graph;
    fl::eval(x.tensor());

    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)); 
    {
        sum_graph = heavy_ops(x);
    }
    check(cudaStreamEndCapture(stream, &graph));
    fl::sync();
    nvtxRangePop();

    check(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // cuda graph execution
    auto start_2 = fl::Timer::start();
    for (auto i = 0; i < iters; ++i) {
        nvtxRangePush("cudaGraph");
        check(cudaGraphLaunch(graphExec, stream));
        fl::sync();
        nvtxRangePop();
    }
    auto graph_ms = fl::Timer::stop(start_2);
    auto sum_val_graph = sum_graph.tensor().scalar<float>();
    
    std::cout << "sum (eager) = " << sum_val_eager << "\n";
    std::cout << "sum (graph) = " << sum_val_graph << "\n";
    std::cout << "Eager time : " << eager_ms << " ms for " << iters << " iters ( "
        << (eager_ms/iters) << " ms/iter)\n";
    std::cout << "Graph time : " << graph_ms << " ms for " << iters << " iters ( "
        << (graph_ms/iters) << " ms/iter)\n";
    std::cout << "end of the process" << std::endl;

    return 0;
} 
