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
  setSeed(0);

  // 랜덤 텐서 (스케일 ↑)
  Tensor X  = rand(dims) * 1e8f;
  Tensor Z  = rand(dims) * 1e8f;
  Tensor Wt = rand(dims) * 1e8f;
  Variable vX(X,false), vZ(Z,false), vW(Wt,false);

  std::vector<float> maxErrs, meanErrs;
  maxErrs.reserve(ITERS); meanErrs.reserve(ITERS);

  // 워밍업
  (vX * vZ + vW).eval();  cudaDeviceSynchronize();

  for (int i = 0; i < ITERS; ++i) {
    // ---------- 경로 2 : FMA ----------
    nvtxRangePushA("path2_fma");
    Variable Y2 = vX * vZ + vW; Y2.eval();
    cudaDeviceSynchronize();
    nvtxRangePop();
  }
}

