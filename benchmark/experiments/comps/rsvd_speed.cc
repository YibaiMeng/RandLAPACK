#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>
#include <lapack.hh>
#include <hamr_buffer.h>

#include <loguru.hpp>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono;

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace RandLAPACK::comps::rsvd;
using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::comps::orth;
using namespace RandLAPACK::comps::rs;
using namespace RandLAPACK::comps::rf;
using namespace RandLAPACK::comps::qb;

namespace RandLAPACK::comps::util {
    Timer profile_timer;
}

class RsvdSpeed : public ::testing::Test
{
protected:
    virtual void SetUp(){};

    virtual void TearDown(){};
    //         A(hamr::buffer_allocator::cpp, m * n, 0.0);
    template <typename T>
    static void generate_random_low_rank_mat(int m, int n, int rank, hamr::buffer<T> &A, int seed)
    {
        hamr::buffer<T> src_1(hamr::buffer_allocator::cpp, m * rank, 0.0), src_2(hamr::buffer_allocator::cpp, rank * n, 0.0);
        RandBLAS::dense_op::gen_rmat_norm<T>(m, rank, src_1.data(), seed);
        RandBLAS::dense_op::gen_rmat_norm<T>(rank, n, src_2.data(), seed + 1);
        blas::gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, rank, 1.0, src_1.data(), m, src_2.data(), rank, 0, A.data(), m);
    }

    template <typename T>
    static long test_rsvd(int64_t m, int64_t n, int64_t actual_rank, int64_t target_rank, uint32_t seed, hamr::buffer<T> &A, int n_subspace_iters = 2)
    {
        LOG_F(INFO, "RSVD on %i by %i matrix", m, n);
        auto alloc = A.get_allocator();
        // Subroutine parameters
        bool verbosity = true;
        bool cond_check = false;
        // Make subroutine objects
        // Stabilization Constructor - Choose default
        Stab<T> Stab(0);
        int64_t passes_per_stablizer = 1;
        // RowSketcher constructor - Choose default (rs1)
        RS<T> RS(Stab, seed, n_subspace_iters, passes_per_stablizer, verbosity, cond_check, 0);
        // Orthogonalization Constructor - Choose CholQR.
        Orth<T> Orth_RF(0);
        // RangeFinder constructor - Choose default (rf1)
        RF<T> RF(RS, Orth_RF, verbosity, cond_check, 0);
        // Orthogonalization Constructor
        RSVD<T> rsvd_obj(
            RS,
            Orth_RF,
            RF);
        int64_t n_oversamples = 5;
        int64_t k = n_oversamples + target_rank;
        LOG_F(INFO, "Actual rank of test input is %i. Target rank %i, number of samples %i", actual_rank, target_rank, k);
        hamr::buffer<T> U(alloc, m * target_rank, 0.0);
        hamr::buffer<T> S(alloc, k, 0.0);
        hamr::buffer<T> VT(alloc, n * n, 0.0);
        lapack::Queue *q = nullptr;
        if (alloc == hamr::buffer_allocator::cuda)
            q = new lapack::Queue(0, 0);
        auto start_rsvd = high_resolution_clock::now();
        cudaDeviceSynchronize();
        profile_timer.start_tag("rsvd");
        rsvd_obj.call(m, n, A, target_rank, n_oversamples, U, S, VT, q);
        if (q)
            q->sync();
        auto stop_rsvd = high_resolution_clock::now();
        profile_timer.accumulate_tag("rsvd");
        long dur_rsvd = duration_cast<microseconds>(stop_rsvd - start_rsvd).count();
        cudaDeviceSynchronize();
        LOG_F(INFO, "RSVD completed in %ld us", dur_rsvd);
        return dur_rsvd;
    }
};

// Test the speed of RSVD for different sized matrices, both for CPU and GPU implementation.
// Also compares against blas::gesvd.
TEST_F(RsvdSpeed, RsvdSpeed)
{

    int seed = 217;
    // Doesn't do anything, just warming up the GPU, as the GPU could be in a power saving / low frequency state.
    LOG_F(WARNING, "It takes dozens of seconds for the GPU to get out of sleep and power up. Warming up before measurement");
    {
        int n = 500;
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n, 0.0);
        int actual_rank = (int)(n * 0.1);
        generate_random_low_rank_mat<double>(n, n, actual_rank, A, 271);
        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        int target_rank = actual_rank + 5;
        long t = test_rsvd<double>(n, n, actual_rank, target_rank, seed, A, 2);
    }

    // Testing square matrices
    std::vector<int> square_matrice_size({100, 500, 1000, 2000, 4000, 5000, 8000});
    std::fstream file;
    // TODO: where to put the file?
    file.open("test_rsvd_speed_square_gpu.dat", std::fstream::app);
    file << "desc"
         << ","
         << "float_precision"
         << ","
         << "width"
         << ","
         << "height"
         << ","
         << "target_rank"
         << ","
         << "subspace_iters"
         << ","
         << "time_us" 
         << ","
         << "device" << std::endl;
    // Test square matrices
    for (int n : square_matrice_size)
    {
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n, 0.0);
        int actual_rank = (int)(n * 0.1);
        int target_rank = actual_rank + 5;
        generate_random_low_rank_mat<double>(n, n, actual_rank, A, seed);

        // Test gesvd
        int m = n;
        hamr::buffer<double> U_gold(hamr::buffer_allocator::cpp, m * m, 0.0);
        hamr::buffer<double> S_gold(hamr::buffer_allocator::cpp, n, 0.0);
        hamr::buffer<double> VT_gold(hamr::buffer_allocator::cpp, n * n, 0.0);
        auto start_svd = high_resolution_clock::now();
        lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, m, n, A.data(), m, S_gold.data(), U_gold.data(), m, VT_gold.data(), n);
        auto stop_svd = high_resolution_clock::now();
        long dur_svd = duration_cast<microseconds>(stop_svd - start_svd).count();
        file << "square"
            << ","
            << "double"
            << "," << n << "," << n << "," << -1 << "," << -1 << "," << dur_svd << ", gesvd CPU" << std::endl;

        // CPU implementation
        for(int cnt = 0; cnt < 2; cnt++) {
            long t = test_rsvd<double>(n, n, actual_rank, target_rank, seed, A, 2);
            cudaDeviceSynchronize();
            file << "square"
                << ","
                << "double"
                << "," << n << "," << n << "," << target_rank << "," << 2 << "," << t << ", CPU" << std::endl;
        }

        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        // GPU implementation
        for(int cnt = 0; cnt < 2; cnt++) {
            long t = test_rsvd<double>(n, n, actual_rank, target_rank, seed, A, 2);
            cudaDeviceSynchronize();
        file << "square"
             << ","
             << "double"
             << "," << n << "," << n << "," << target_rank << "," << 2 << "," << t << ", GPU" << std::endl;
        }
    }
    // Test rectangular matrices
    for (int n : square_matrice_size)
    {
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n / 2, 0.0);
        int actual_rank = (int)(n * 0.1);
        generate_random_low_rank_mat<double>(n, n / 2, actual_rank, A, seed);
        int target_rank = actual_rank + 5;
        // CPU implementation
        for(int cnt = 0; cnt < 2; cnt++) {
            long t = test_rsvd<double>(n, n / 2, actual_rank, target_rank, seed, A, 2);
            file << "rectangle"
                << ","
                << "double"
                << "," << n << "," << n / 2 << "," << target_rank << "," << 2 << "," << t << ", CPU" << std::endl;
        }

        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        // GPU implementation
        for(int cnt = 0; cnt < 2; cnt++) {
            long t = test_rsvd<double>(n, n / 2, actual_rank, target_rank, seed, A, 2);
            cudaDeviceSynchronize();
        file << "rectangle"
             << ","
             << "double"
            << "," << n << "," << n / 2 << "," << target_rank << "," << 2 << "," << ", GPU" << std::endl;
        }
    }
    file.close();
}

TEST_F(RsvdSpeed, Profile)
{
    // Doesn't do anything, just warming up the GPU, as the GPU could be in a power saving / low frequency state.
    LOG_F(WARNING, "It takes dozens of seconds for the GPU to get out of sleep and power up. Warming up before measurement");
    {
        int n = 500;
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n, 0.0);
        int actual_rank = (int)(n * 0.1);
        generate_random_low_rank_mat<double>(n, n, actual_rank, A, 271);
        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        int target_rank = actual_rank + 5;
        long t = test_rsvd<double>(n, n, actual_rank, target_rank, 271, A, 2);
    }
    // Enables verbose logging for loguru. See its doc for more detail.
    loguru::g_stderr_verbosity = 1;
    // Testing square matrices
    int n = 8000;
    hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n, 0.0);
    int actual_rank = (int)(n * 0.1);
    generate_random_low_rank_mat<double>(n, n, actual_rank, A, 271);
    CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
    A.synchronize();
    int target_rank = actual_rank + 5;
    profile_timer.clear();
    long t = test_rsvd<double>(n, n, actual_rank, target_rank, 271, A, 2);
    cudaDeviceSynchronize();
    profile_timer.print();
    loguru::g_stderr_verbosity = 0;
}


// Test the runtime of different subspace iteration settings.
TEST_F(RsvdSpeed, SubspaceIteration)
{
    // Doesn't do anything, just warming up the GPU, as the GPU could be in a power saving / low frequency state.
    LOG_F(WARNING, "It takes dozens of seconds for the GPU to get out of sleep and power up. Warming up before measurement");
    {
        int n = 500;
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n, 0.0);
        int actual_rank = (int)(n * 0.1);
        generate_random_low_rank_mat<double>(n, n, actual_rank, A, 271);
        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        int target_rank = actual_rank + 5;
        long t = test_rsvd<double>(n, n, actual_rank, target_rank, 271, A, 2);
    }
    loguru::g_stderr_verbosity = 1;
    // Testing square matrices
    int n = 8000;
    hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n, 0.0);
    int actual_rank = (int)(n * 0.1);
    generate_random_low_rank_mat<double>(n, n, actual_rank, A, 271);
    CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
    A.synchronize();

    for(int subspace_iter : {0, 1, 2, 3, 4, 5}) {
        profile_timer.clear();
        int target_rank = actual_rank + 5;
        long t = test_rsvd<double>(n, n, actual_rank, target_rank, 271, A, subspace_iter);
        cudaDeviceSynchronize();
        profile_timer.print();
    }
    loguru::g_stderr_verbosity = 0;
}

// Microbenchmark to check GEMM speed of blaspp's chosen CPU implementation, 
// and blaspp's cuBLAS wrapper.
TEST_F(RsvdSpeed, GemmMicrobench)
{
        int m = 8000, n = 8000, k = 8000;
        hamr::buffer<double> src_1(hamr::buffer_allocator::cpp, m * k, 0.0), src_2(hamr::buffer_allocator::cpp, k * n, 0.0);
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, m * n, 0.0);
        int seed = 271;
        RandBLAS::dense_op::gen_rmat_norm<double>(m, k, src_1.data(), seed);
        RandBLAS::dense_op::gen_rmat_norm<double>(k, n, src_2.data(), seed + 1);
        {
        auto start_gemm = high_resolution_clock::now();
        blas::gemm<double>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, 1.0, src_1.data(), m, src_2.data(), k, 0, A.data(), m);
        auto stop_gemm = high_resolution_clock::now();
        long dur_gemm = duration_cast<microseconds>(stop_gemm - start_gemm).count();
        LOG_F(INFO, "mkl BLAS: %ld us", dur_gemm);
        }
        CHECK_F(src_1.move(hamr::buffer_allocator::cuda) == 0);
        CHECK_F(src_2.move(hamr::buffer_allocator::cuda) == 0);
        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        src_1.synchronize();
        src_2.synchronize();
        A.synchronize();
        blas::Queue q(0,0);
        {
        auto start_gemm = high_resolution_clock::now();
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, 1.0, src_1.data(), m, src_2.data(), k, 0, A.data(), m, q);
        q.sync();
        auto stop_gemm = high_resolution_clock::now();
        long dur_gemm = duration_cast<microseconds>(stop_gemm - start_gemm).count();
        LOG_F(INFO, "cuBLAS: %ld us", dur_gemm);
        }
}