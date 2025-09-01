#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <string>

class GpuTimer {
public:
    explicit GpuTimer(std::string name = "", cudaStream_t s = 0)
        : m_name{name}, m_stream{s}
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }
    ~GpuTimer() noexcept
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

    void start()
    {
        cudaEventRecord(m_start, m_stream);
    }

    void stop()
    {
        cudaEventRecord(m_stop,  m_stream);
    }

    float elapsedMilliseconds(const int iter) const
    {
        cudaEventSynchronize(m_stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, m_start, m_stop);
        return ms/static_cast<float>(iter);
    }

    void printElapsedTime(const int iter, const char* fmt = "%s elapsed: %.3f ms\n") const
    {
        printf(fmt, m_name.empty() ? "(unnamed)" : m_name.data(),
               elapsedMilliseconds(iter));
    }

    void reset() { start(); }

private:
    std::string m_name;
    cudaStream_t m_stream{0};
    cudaEvent_t  m_start{}, m_stop{};
};

