/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    unittest.cpp

Abstract:

    This module implements unit tests of the MLAS library.

--*/

#include <stdio.h>
#include <memory.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <chrono>

#include <mlas.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

class MatrixGuardBuffer
{
public:
    MatrixGuardBuffer()
    {
        _BaseBuffer = nullptr;
        _BaseBufferSize = 0;
        _ElementsAllocated = 0;
    }

    ~MatrixGuardBuffer(void)
    {
        ReleaseBuffer();
    }

    float* GetBuffer(size_t Elements)
    {
        //
        // Check if the internal buffer needs to be reallocated.
        //

        if (Elements > _ElementsAllocated) {

            ReleaseBuffer();

            //
            // Reserve a virtual address range for the allocation plus an unmapped
            // guard region.
            //

            constexpr size_t BufferAlignment = 64 * 1024;
            constexpr size_t GuardPadding = 256 * 1024;

            size_t BytesToAllocate = ((Elements * sizeof(float)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

            _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
            _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
            _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

            if (_BaseBuffer == nullptr) {
                throw std::bad_alloc();
            }

            //
            // Commit the number of bytes for the allocation leaving the upper
            // guard region as unmapped.
            //

#if defined(_WIN32)
            if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
                throw std::bad_alloc();
            }
#else
            if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
                throw std::bad_alloc();
            }
#endif

            _ElementsAllocated = BytesToAllocate / sizeof(float);
            _GuardAddress = (float*)((unsigned char*)_BaseBuffer + BytesToAllocate);
        }

        //
        //
        //

        float* GuardAddress = _GuardAddress;
        float* buffer = GuardAddress - Elements;

        const int MinimumFillValue = -23;
        const int MaximumFillValue = 23;

        int FillValue = MinimumFillValue;
        float* FillAddress = buffer;

        while (FillAddress < GuardAddress) {

            *FillAddress++ = (float)FillValue;

            FillValue++;

            if (FillValue > MaximumFillValue) {
                FillValue = MinimumFillValue;
            }
        }

        return buffer;
    }

    void ReleaseBuffer(void)
    {
        if (_BaseBuffer != nullptr) {

#if defined(_WIN32)
            VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
            munmap(_BaseBuffer, _BaseBufferSize);
#endif

            _BaseBuffer = nullptr;
            _BaseBufferSize = 0;
        }

        _ElementsAllocated = 0;
    }

private:
    size_t _ElementsAllocated;
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    float* _GuardAddress;
};

class MlasTestBase
{
public:
    virtual
    ~MlasTestBase(
        void
        )
    {
    }

    //
    // Contains tests that run quickly as part of a checkin integration to
    // sanity check that the functionality is working.
    //

    virtual
    void
    ExecuteShort(
        void
        ) = 0;

    //
    // Contains tests that can run slowly to more exhaustively test that
    // functionality is working across a broader range of parameters.
    //

    virtual
    void
    ExecuteLong(
        void
        ) = 0;
};

class MlasSgemmTiming : public MlasTestBase
{
private:
    void
    Timing(
        int numIter,
        int warmupCount,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        float beta
        )
    {
        const float* A = BufferA.GetBuffer(K * M);
        const float* B = BufferB.GetBuffer(N * K);
        float* C = BufferC.GetBuffer(N * M);
        float* CReference = BufferCReference.GetBuffer(N * M);

        Timing(numIter, warmupCount, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, CReference, N);
        // Timing(CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, CReference, N);
        // Timing(CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, CReference, N);
        // Timing(CblasTrans, CblasTrans, M, N, K, alpha, A, M, B, K, beta, C, CReference, N);
    }

    void
    Timing(
        int numIter,
        int warmupCount,
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const float* A,
        size_t lda,
        const float* B,
        size_t ldb,
        float beta,
        float* C,
        float* CReference,
        size_t ldc
        )
    {
        std::fill_n(C, M * N, -0.5f);
        std::fill_n(CReference, M * N, -0.5f);

        for(int i = 0; i < warmupCount; ++i)
        {
            MlasSgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, nullptr);
        }

        using clock_t = std::chrono::high_resolution_clock;
        using second_t = std::chrono::duration<double, std::ratio<1> >;
        std::chrono::time_point<clock_t> start = clock_t::now();
        for(int i = 0; i < numIter; ++i)
        {
            MlasSgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, nullptr);
        }
        auto elapsed = std::chrono::duration_cast<second_t>(clock_t::now() - start).count();
        auto timePerIter = elapsed / numIter;

        printf("time=%f, iter=%d, time_per_iter=%f\n", elapsed, numIter, timePerIter);

        // verify
        ReferenceSgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, CReference, ldc);

        for (size_t f = 0; f < M * N; f++) {
            // Sensitive to comparing positive/negative zero.
            if (C[f] != CReference[f]) {
                printf("mismatch TransA=%d, TransB=%d, M=%zd, N=%zd, K=%zd, alpha=%f, beta=%f!\n", TransA, TransB, M, N, K, alpha, beta);
            }
        }
    }

    void
    ReferenceSgemm(
        CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB,
        size_t M,
        size_t N,
        size_t K,
        float alpha,
        const float* A,
        size_t lda,
        const float* B,
        size_t ldb,
        float beta,
        float* C,
        size_t ldc
        )
    {
        if (TransA == CblasNoTrans) {

            if (TransB == CblasNoTrans) {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const float* a = A + (m * lda);
                        const float* b = B + n;
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += ldb;
                            a += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }

            } else {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const float* a = A + (m * lda);
                        const float* b = B + (n * ldb);
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += 1;
                            a += 1;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            }

        } else {

            if (TransB == CblasNoTrans) {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const float* a = A + m;
                        const float* b = B + n;
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += ldb;
                            a += lda;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }

            } else {

                for (size_t m = 0; m < M; m++) {

                    for (size_t n = 0; n < N; n++) {

                        const float* a = A + m;
                        const float* b = B + (n * ldb);
                        float* c = C + (m * ldc) + n;
                        float sum = 0.0f;

                        for (size_t k = 0; k < K; k++) {
                            sum += (*b * *a);
                            b += 1;
                            a += lda;
                        }

                        *c = (*c * beta) + (sum * alpha);
                    }
                }
            }
        }
    }

    MatrixGuardBuffer BufferA;
    MatrixGuardBuffer BufferB;
    MatrixGuardBuffer BufferC;
    MatrixGuardBuffer BufferCReference;

public:
    void
    ExecuteShort(
        void
        ) override
    {
        const int numIter = 1000;
        const int warmupCount = 100;
        const int size = 256;
        Timing(numIter, warmupCount, size, size, size, 1.0f, 0.0f);
    }

    void
    ExecuteLong(
        void
        ) override
    {
    }
};

int
#if defined(_WIN32)
__cdecl
#endif
main(
    void
    )
{
    printf("SGEMM timing.\n");
    std::make_unique<MlasSgemmTiming>()->ExecuteShort();

    printf("Done.\n");

    return 0;
}
