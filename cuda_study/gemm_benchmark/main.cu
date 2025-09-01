#include <iostream>
#include <vector>
#include <type_traits>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/arch/arch.h>

#include "gpuTimer.h"

void compare(const std::vector<float>& test_object
    , const std::vector<float>& target
    )
{
    auto max_abs_diff = 0.0f;
    for (auto i = 0u; i < test_object.size(); ++i) {    
        auto diff = std::abs(test_object[i] - target[i]);
        if (diff > max_abs_diff) { max_abs_diff = diff; }
    }  
    
    std::cout << "Max abs diff = " << max_abs_diff << "\n";
}

void gemm_cublasLt(int M, int N, int K, int batch
    , const std::vector<float>& hA, int lda, int strideA
    , const std::vector<float>& hB, int ldb, int strideB
    , const std::vector<float>& hC_ans, int ldc, int strideC
    , int iter
    , bool use_tf32
    )
{
    auto name = std::stringstream();
    name << "gemm_cublasLt_" << M << "_" << N << "_" << K << "_" << batch << (use_tf32? "_tf32" : "_fp32");
    auto timer = GpuTimer(name.str());

    using Element = float;
    
    size_t sizeA = hA.size(), sizeB = hB.size(), sizeC = hC_ans.size();
    auto sA = static_cast<int64_t>(strideA);
    auto sB = static_cast<int64_t>(strideB);
    auto sC = static_cast<int64_t>(strideC);

    Element* dA;
    Element* dB;
    Element* dC;

    cudaMalloc(&dA, sizeA*sizeof(Element));
    cudaMalloc(&dB, sizeA*sizeof(Element));
    cudaMalloc(&dC, sizeA*sizeof(Element));

    cudaMemcpy(dA, hA.data(), sizeA*sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeB*sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sizeC*sizeof(Element));

    cublasLtHandle_t lt;
    cublasLtCreate(&lt);

    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    if (use_tf32) {
        compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    }

    cublasLtMatmulDesc_t op_desc;
    cublasLtMatmulDescCreate(&op_desc
        , compute
        , CUDA_R_32F
        );

    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, ldc);
    
    cublasLtMatrixLayoutSetAttribute(Adesc
        , CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
        , &batch, sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(Bdesc
        , CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
        , &batch, sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(Cdesc
        , CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
        , &batch, sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(Adesc
        , CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
        , &sA, sizeof(sA));
    cublasLtMatrixLayoutSetAttribute(Bdesc
        , CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
        , &sB, sizeof(sB));
    cublasLtMatrixLayoutSetAttribute(Cdesc
        , CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
        , &sC, sizeof(sC));

    float alpha = 1.f;
    float beta = 0.f;

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);

    size_t workspace_size = 32<<20; //32 MB 
    void* workspace;
    cudaMalloc(&workspace, workspace_size);
    cublasLtMatmulPreferenceSetAttribute(pref
        , CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
        , &workspace_size, sizeof(workspace_size));

    constexpr int MAX_CAND = 16;
    std::vector<cublasLtMatmulHeuristicResult_t> results(MAX_CAND);
    int returned = 0;

    cublasLtMatmulAlgoGetHeuristic(lt, op_desc
        , Adesc, Bdesc, Cdesc, Cdesc
        , pref, MAX_CAND, results.data(), &returned);
    
    if (returned == 0) {
        std::cerr << "gemm_cublaslt: no suitable algos" << std::endl;
    }

    cublasLtMatmul(lt, op_desc
        , &alpha, dA, Adesc
                , dB, Bdesc 
        , &beta , dC, Cdesc
                , dC, Cdesc
        , &results[0].algo
        , workspace
        , workspace_size
        , 0
        );

    timer.start();
    for (auto i = 0; i < iter; ++i) {
        cublasLtMatmul(lt, op_desc
            , &alpha, dA, Adesc
                    , dB, Bdesc 
            , &beta , dC, Cdesc
                    , dC, Cdesc
            , &results[0].algo
            , workspace
            , workspace_size
            , 0
            );
    }
    timer.stop();
    timer.printElapsedTime(iter);

    std::vector<float> hC(sizeC);
    cudaMemcpy(hC.data(), dC, sizeC*sizeof(Element), cudaMemcpyDeviceToHost);
    
    compare(hC, hC_ans);

    cublasLtDestroy(lt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void gemm_cublas(int M, int N, int K, int batch
    , const std::vector<float>& hA, int lda, int strideA
    , const std::vector<float>& hB, int ldb, int strideB
    , const std::vector<float>& hC_ans, int ldc, int strideC
    , int iter
    , bool use_tf32
    )
{
    auto name = std::stringstream();
    name << "gemm_cublas_" << M << "_" << N << "_" << K << "_" << batch << (use_tf32? "_tf32" : "_fp32");
    auto timer = GpuTimer(name.str());

    using Element = float;
    size_t sizeA = hA.size();
    size_t sizeB = hB.size();
    size_t sizeC = hC_ans.size();

    Element *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA*sizeof(Element));
    cudaMalloc(&dB, sizeB*sizeof(Element));
    cudaMalloc(&dC, sizeC*sizeof(Element));

    cudaMemcpy(dA, hA.data(), sizeA*sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), sizeB*sizeof(Element), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.f;
    float beta = 0.f;

    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
    if (use_tf32) {
        compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    }

    cublasGemmStridedBatchedEx(
        handle
        , CUBLAS_OP_N
        , CUBLAS_OP_N
        , M, N, K
        , &alpha
        , dA, CUDA_R_32F, lda, strideA
        , dB, CUDA_R_32F, ldb, strideB
        , &beta
        , dC, CUDA_R_32F, ldc, strideC
        , batch
        , compute
        , CUBLAS_GEMM_DEFAULT
    );

    timer.start();
    for (int i = 0; i < iter; ++i) {
        cublasGemmStridedBatchedEx(
            handle
            , CUBLAS_OP_N
            , CUBLAS_OP_N
            , M, N, K
            , &alpha
            , dA, CUDA_R_32F, lda, strideA
            , dB, CUDA_R_32F, ldb, strideB
            , &beta
            , dC, CUDA_R_32F, ldc, strideC
            , batch
            , compute
            , CUBLAS_GEMM_DEFAULT
        );
    }
    timer.stop();
    timer.printElapsedTime(iter);

    std::vector<float> hC(sizeC);
    cudaMemcpy(hC.data(), dC, sizeC*sizeof(Element), cudaMemcpyDeviceToHost);
    
    compare(hC, hC_ans);

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(st) \
  do { auto _s = (st); if (_s != cutlass::Status::kSuccess) { \
    fprintf(stderr, "CUTLASS error %d at %s:%d\n", int(_s), __FILE__, __LINE__); \
    return; } } while(0)
#endif

template<bool USE_TF32>
struct CutlassGemmTypes {
  using ElemA = std::conditional_t<USE_TF32, cutlass::tfloat32_t, float>;
  using ElemB = std::conditional_t<USE_TF32, cutlass::tfloat32_t, float>;
  using ElemC = float;
  using Acc   = float;

  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using OpClass = std::conditional_t<USE_TF32,
                    cutlass::arch::OpClassTensorOp,
                    cutlass::arch::OpClassSimt>;
  using ArchTag = cutlass::arch::Sm80;

  using ThreadBlockShape = std::conditional_t<USE_TF32,
                          cutlass::gemm::GemmShape<128,128,64>,
                          cutlass::gemm::GemmShape<128,128, 8>>;
  using WarpShape        = std::conditional_t<USE_TF32,
                          cutlass::gemm::GemmShape< 64, 64,64>,
                          cutlass::gemm::GemmShape< 64, 64, 8>>;
  using InstructionShape = std::conditional_t<USE_TF32,
                          cutlass::gemm::GemmShape< 16,  8, 8>,
                          cutlass::gemm::GemmShape<  1,  1, 1>>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElemC, 4, Acc, Acc>;
  using Swizzle    = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  static constexpr int kStages = USE_TF32 ? 3 : 2;

  // 정렬(벡터화) 요구: TF32는 128-bit 벡터화 위해 4, FP32는 1
  static constexpr int kAlignA = USE_TF32 ? 4 : 1;
  static constexpr int kAlignB = USE_TF32 ? 4 : 1;

  using Gemm = cutlass::gemm::device::Gemm<
      ElemA, LayoutA,
      ElemB, LayoutB,
      ElemC, LayoutC,
      Acc, OpClass, ArchTag,
      ThreadBlockShape, WarpShape, InstructionShape,
      EpilogueOp, Swizzle, kStages,
      /*AlignmentA=*/kAlignA,
      /*AlignmentB=*/kAlignB>;
};

inline bool tf32_aligned(int lda, int ldb, int K) {
  // 안전하게 8의 배수 요구 (tfloat32 텐서오프 타일과 정렬 맞춤)
  return (lda % 8 == 0) && (ldb % 8 == 0) && (K % 8 == 0);
}

void gemm_cutlass_v2(int M, int N, int K, int batch,
    const std::vector<float>& hA, int lda, int strideA,
    const std::vector<float>& hB, int ldb, int strideB,
    const std::vector<float>& hC_ans, int ldc, int strideC,
    int iter,
    bool use_tf32)
{
  // 타이머(네가 쓰던 GpuTimer 가정)
  std::stringstream ss;
  ss << "gemm_cutlass_v2_" << M << "_" << N << "_" << K << "_" << batch
     << (use_tf32 ? "_tf32" : "_fp32");
  GpuTimer timer(ss.str());

  using GemmFP32 = typename CutlassGemmTypes<false>::Gemm;
  using GemmTF32 = typename CutlassGemmTypes<true >::Gemm;

  // 출력 버퍼
  std::vector<float> hC((long long)strideC * batch, 0.f);

  // 디바이스 C
  float* dC = nullptr;
  cudaMalloc(&dC, sizeof(float) * hC.size());
  cudaMemset(dC, 0, sizeof(float) * hC.size());

  if (!use_tf32) {
    // ---- FP32 SIMT 경로 ----
    float *dA_f = nullptr, *dB_f = nullptr;
    cudaMalloc(&dA_f, sizeof(float) * hA.size());
    cudaMalloc(&dB_f, sizeof(float) * hB.size());
    cudaMemcpy(dA_f, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_f, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice);

    GemmFP32 gemm;
    typename GemmFP32::Arguments args{
      {M, N, K},
      (int64_t)batch,
      {1.0f, 0.0f},
      {dA_f, lda}, {dB_f, ldb},
      {dC,   ldc}, {dC,   ldc},
      (int64_t)strideA, (int64_t)strideB, (int64_t)strideC, (int64_t)strideC
    };

    CUTLASS_CHECK(gemm.initialize(args));
    CUTLASS_CHECK(gemm()); // warm-up

    timer.start();
    for (int i = 0; i < iter; ++i) {
      CUTLASS_CHECK(gemm());
    }
    timer.stop();
    timer.printElapsedTime(iter);

    cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost);
    cudaFree(dA_f); cudaFree(dB_f);
  } else {
    // ---- TF32 TensorOp 경로 ----
    if (!tf32_aligned(lda, ldb, K)) {
      fprintf(stderr,
        "[CUTLASS][TF32] require K, lda, ldb to be multiples of 8. "
        "Given K=%d lda=%d ldb=%d -> falling back to FP32.\n", K, lda, ldb);
      // FP32로 폴백
      using Gemm = GemmFP32;
      float *dA_f = nullptr, *dB_f = nullptr;
      cudaMalloc(&dA_f, sizeof(float) * hA.size());
      cudaMalloc(&dB_f, sizeof(float) * hB.size());
      cudaMemcpy(dA_f, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dB_f, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice);

      Gemm gemm;
      typename Gemm::Arguments args{
        {M, N, K},
        (int64_t)batch,
        {1.0f, 0.0f},
        {dA_f, lda}, {dB_f, ldb},
        {dC,   ldc}, {dC,   ldc},
        (int64_t)strideA, (int64_t)strideB, (int64_t)strideC, (int64_t)strideC
      };
      CUTLASS_CHECK(gemm.initialize(args));
      CUTLASS_CHECK(gemm());
      timer.start();
      for (int i = 0; i < iter; ++i) CUTLASS_CHECK(gemm());
      timer.stop();
      timer.printElapsedTime(iter);
      cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost);
      cudaFree(dA_f); cudaFree(dB_f);
    } else {
      // TF32로 변환해서 업로드
      using ElemTF = cutlass::tfloat32_t;
      std::vector<ElemTF> hA_tf(hA.size()), hB_tf(hB.size());
      cutlass::NumericConverter<ElemTF, float> to_tf32;
      for (size_t i=0;i<hA.size();++i) hA_tf[i] = to_tf32(hA[i]);
      for (size_t i=0;i<hB.size();++i) hB_tf[i] = to_tf32(hB[i]);

      ElemTF *dA_tf = nullptr, *dB_tf = nullptr;
      cudaMalloc(&dA_tf, sizeof(ElemTF) * hA_tf.size());
      cudaMalloc(&dB_tf, sizeof(ElemTF) * hB_tf.size());
      cudaMemcpy(dA_tf, hA_tf.data(), sizeof(ElemTF) * hA_tf.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(dB_tf, hB_tf.data(), sizeof(ElemTF) * hB_tf.size(), cudaMemcpyHostToDevice);

      GemmTF32 gemm;
      typename GemmTF32::Arguments args{
        {M, N, K},
        (int64_t)batch,
        {1.0f, 0.0f},
        {dA_tf, lda}, {dB_tf, ldb},
        {dC,    ldc}, {dC,    ldc},
        (int64_t)strideA, (int64_t)strideB, (int64_t)strideC, (int64_t)strideC
      };

      CUTLASS_CHECK(gemm.initialize(args));
      CUTLASS_CHECK(gemm()); // warm-up

      timer.start();
      for (int i=0;i<iter;++i) {
        CUTLASS_CHECK(gemm());
      }
      timer.stop();
      timer.printElapsedTime(iter);

      cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost);
      cudaFree(dA_tf); cudaFree(dB_tf);
    }
  }

  // 결과 비교 (네가 가진 compare 사용)
  compare(hC, hC_ans);

  cudaFree(dC);
}

void gemm_cutlass(int M, int N, int K, int batch
    , const std::vector<float>& hA, int lda, int strideA
    , const std::vector<float>& hB, int ldb, int strideB
    , const std::vector<float>& hC_ans, int ldc, int strideC
    , int iter
    , bool use_tf32
    ) 
{
    auto name = std::stringstream();
    name << "gemm_cutlass_" << M << "_" << N << "_" << K << "_" << batch << (use_tf32? "_tf32" : "_fp32");
    auto timer = GpuTimer(name.str());

//    using ElementA = float; //cutlass::tfloat32_t
//    using ElementB = float; //cutlass::tfloat32_t
    using ElementC = float;
    using ElementAcc = float;
    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
//    using OpClass = cutlass::arch::OpClassSimt;  // SIMT or TensorOp
//    using SmArch = cutlass::arch::Sm80;

    auto hC = std::vector<ElementC>(batch * strideC);

    auto run_path = [&](auto tag) {
        using ElemA = std::conditional_t<decltype(tag)::value
            , cutlass::tfloat32_t, float>;
        using ElemB = std::conditional_t<decltype(tag)::value
            , cutlass::tfloat32_t, float>;
        using GemmBatched = cutlass::gemm::device::GemmBatched<
            ElemA, LayoutA
            , ElemB, LayoutB
            , ElementC, LayoutC
            >;

        ElemA* dA = nullptr; 
        ElemB* dB = nullptr;
        ElementC* dC =nullptr;
        cudaMalloc(&dA, sizeof(ElemA)   * hA.size());
        cudaMalloc(&dB, sizeof(ElemB)   * hB.size());
        cudaMalloc(&dC, sizeof(ElementC)* hC.size());

        if constexpr (std::is_same_v<ElemA, cutlass::tfloat32_t>) {
            std::vector<ElemA> hA_tf(hA.size());
            std::vector<ElemB> hB_tf(hB.size());
            cutlass::NumericConverter<ElemA, float> to_tf32;
            for (size_t i = 0; i < hA.size(); ++i) {
                hA_tf[i] = to_tf32(hA[i]);
            }
            for (size_t i = 0; i < hB.size(); ++i) {
                hB_tf[i] = to_tf32(hB[i]);
            }

            cudaMemcpy(dA, hA_tf.data(), sizeof(ElemA) * hA_tf.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hB_tf.data(), sizeof(ElemB) * hB_tf.size(), cudaMemcpyHostToDevice);
        }
        else {
            cudaMemcpy(dA, hA.data(), sizeof(ElemA) * hA.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hB.data(), sizeof(ElemB) * hB.size(), cudaMemcpyHostToDevice);
        }

        ElementAcc alpha = ElementAcc(1);
        ElementAcc beta  = ElementAcc(0);

        typename GemmBatched::Arguments args(
            {M, N, K}
            , {dA, lda}, strideA
            , {dB, ldb}, strideB
            , {dC, ldc}, strideC
            , {dC, ldc}, strideC
            , {alpha, beta}
            , batch
            );
    
        GemmBatched gemm;
    
        // warm-up
        gemm(args);
    
        // measurements
        auto status = cutlass::Status{};
    
        timer.start();
        for (auto i = 0; i < iter; ++i) {
            status = gemm(args);
        }
        timer.stop();
        timer.printElapsedTime(iter);
    
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GemmBatched failed!: " << int(status) << "\n";
            return;
        }
    
        cudaMemcpy(hC.data(), dC, sizeof(ElementC)*hC.size(), cudaMemcpyDeviceToHost); 
    
        compare(hC, hC_ans);
    
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }; // run_path
    
    if (use_tf32) { run_path(std::true_type{}); }
    else { run_path(std::false_type{}); }
}

struct GemmArgs {
    int M = 128;
    int N = 128;
    int K = 128;
    int batch = 1;
};

GemmArgs parse_args(int argc, char** argv) {
    GemmArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        auto next = [&](int& var) {
            if (i + 1 < argc) {
                var = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for " << s << "\n";
                std::exit(1);
            }
        };
        if (s == "--m") next(args.M);
        else if (s == "--n") next(args.N);
        else if (s == "--k") next(args.K);
        else if (s == "--batch") next(args.batch);
        else if (s == "--help" || s == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--m M] [--n N] [--k K] [--batch B]\n";
            std::exit(0);
        }
    }
    return args;
}

int main(int argc, char** argv) 
{
    GemmArgs args = parse_args(argc, argv);

    std::cout << "gemm benchmark: " << "M=" << args.M
              << " N=" << args.N
              << " K=" << args.K
              << " batch=" << args.batch << "\n";

    // A[M x K] * B[K x N] = C[M x N]
    int M = args.M;
    int N = args.N; 
    int K = args.K;
    int batch = args.batch;

    int lda = M; int ldb = K; int ldc = M;

    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;

    auto hA = std::vector<float>(strideA * batch);
    auto hB = std::vector<float>(strideB * batch);
    auto hC = std::vector<float>(strideC * batch);

    for (auto b = 0; b < batch; ++b) {
        auto* A = hA.data() + b*strideA;
        auto* B = hB.data() + b*strideB;
        for (int j = 0; j < K; ++j) {
            for (int i = 0; i < M; ++i) {
                A[i + j*lda] = float(((i + j + b)%7)-3);
            }
        }
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < K; ++i) {
                B[i + j*ldb] = float(((i - j + b)%5));
            }
        }
    }

    for (auto b = 0; b < batch; ++b) {
        auto* A = hA.data() + b * strideA;
        auto* B = hB.data() + b * strideB;
        auto* C = hC.data() + b * strideC;

        for (auto j = 0; j < N; ++j) {
            for (auto i = 0; i < M; ++i) {
                C[i + j*ldc] = 0.f;
                for (auto k = 0; k < K; ++k) {
                    C[i + j*ldc] += A[i + k * lda]*B[k + j * ldb];
                }
            }
        }
    }

    gemm_cutlass(M, N, K, batch
        , hA, lda, strideA
        , hB, ldb, strideB
        , hC, ldc, strideC
        , 10
        , false
        );
    gemm_cutlass(M, N, K, batch
        , hA, lda, strideA
        , hB, ldb, strideB
        , hC, ldc, strideC
        , 10
        , true
        );
    gemm_cublas(M, N, K, batch
        , hA, lda, strideA
        , hB, ldb, strideB
        , hC, ldc, strideC
        , 10
        , false
        );
    gemm_cublas(M, N, K, batch
        , hA, lda, strideA
        , hB, ldb, strideB
        , hC, ldc, strideC
        , 10
        , true
        );
    gemm_cublasLt(M, N, K, batch
        , hA, lda, strideA
        , hB, ldb, strideB
        , hC, ldc, strideC
        , 10
        , false
        );
    gemm_cublasLt(M, N, K, batch
        , hA, lda, strideA
        , hB, ldb, strideB
        , hC, ldc, strideC
        , 10
        , true
        );
}
