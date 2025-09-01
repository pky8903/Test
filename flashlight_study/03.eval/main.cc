// main.cc
#include <flashlight/fl/flashlight.h>
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>

using namespace fl;

int main() {
  constexpr int ITERS = 100;
  const int H = 4096, W = 4096;
  Shape dims{H, W};

  std::vector<float> maxErrs, meanErrs;
  maxErrs.reserve(ITERS); meanErrs.reserve(ITERS);

  for (int i = 0; i < ITERS; ++i) {
    setSeed(i);
    
    Tensor X  = rand(dims) * 1e1f;
    Tensor Z  = rand(dims) * 1e1f;
    Tensor Wt = rand(dims) * 1e1f;
    Variable vX(X,false), vZ(Z,false), vW(Wt,false);
    (vX * vZ + vW).eval();  cudaDeviceSynchronize();

    nvtxRangePushA("path1_eval_mid");
    Variable t = vX * vZ; t.eval();
    Variable Y1 = t + vW; Y1.eval();
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("path2_fma");
    Variable Y2 = vX * vZ + vW; Y2.eval();
    cudaDeviceSynchronize();
    nvtxRangePop();

    Tensor diff = abs(Y1.tensor() - Y2.tensor());
    auto maxerr = amax(diff).scalar<float>();
    auto meanerr = mean(diff).scalar<float>();
    std::cout << i << "-iter, max err: " << maxerr << ", mean err: " << meanerr << std::endl;
    maxErrs.push_back(maxerr);
    meanErrs.push_back(meanerr);
  }

  auto avg = [](const auto& v){
    return std::accumulate(v.begin(), v.end(), 0.f) / v.size();
  };
  std::cout << "iters=" << ITERS
            << "  avg max|Δ|="  << avg(maxErrs)
            << "  avg mean|Δ|=" << avg(meanErrs) << '\n';
}

