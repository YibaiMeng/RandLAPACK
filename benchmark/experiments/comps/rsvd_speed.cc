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
        int64_t passes_per_stablizer = 5;
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
        rsvd_obj.call(m, n, A, target_rank, n_oversamples, n_subspace_iters, U, S, VT, q);
        if (q)
            q->sync();
        auto stop_rsvd = high_resolution_clock::now();
        long dur_rsvd = duration_cast<microseconds>(stop_rsvd - start_rsvd).count();
        LOG_F(INFO, "RSVD completed in %ld us", dur_rsvd);
        return dur_rsvd;
    }
};

TEST_F(RsvdSpeed, RsvdSpeed)
{

    // Generating various matrices

    // hamr::buffer<T> A(alloc, m * n, 0.0);
    // gen_mat_type<T>(m, n, A, k, seed, mat_type);

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
         << "time_us" << std::endl;

    for (int n : square_matrice_size)
    {
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n, 0.0);
        int actual_rank = (int)(n * 0.1);
        generate_random_low_rank_mat<double>(n, n, actual_rank, A, 271);
        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        int target_rank = actual_rank + 5;
        long t = test_rsvd<double>(n, n, actual_rank, actual_rank + 5, 271, A, 2);
        file << "square"
             << ","
             << "double"
             << "," << n << "," << n << "," << target_rank << "," << 2 << "," << t << std::endl;
        t = test_rsvd<double>(n, n, actual_rank, actual_rank + 5, 271, A, 5);
        file << "square"
             << ","
             << "double"
             << "," << n << "," << n << "," << target_rank << "," << 5 << "," << t << std::endl;
    }

    for (int n : square_matrice_size)
    {
        hamr::buffer<double> A(hamr::buffer_allocator::cpp, n * n / 2, 0.0);
        int actual_rank = (int)(n * 0.1);
        generate_random_low_rank_mat<double>(n, n / 2, actual_rank, A, 271);
        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        int target_rank = actual_rank + 5;
        long t = test_rsvd<double>(n, n / 2, actual_rank, actual_rank + 5, 271, A, 2);
        file << "rectangle"
             << ","
             << "double"
             << "," << n << "," << n / 2 << "," << target_rank << "," << 2 << "," << t << std::endl;
        t = test_rsvd<double>(n, n / 2, actual_rank, actual_rank + 5, 271, A, 5);
        file << "rectangle"
             << ","
             << "double"
             << "," << n << "," << n / 2 << "," << target_rank << "," << 5 << "," << t << std::endl;
    }

    for (int n : square_matrice_size)
    {
        hamr::buffer<float> A(hamr::buffer_allocator::cpp, n * n, 0.0);
        int actual_rank = (int)(n * 0.1);
        generate_random_low_rank_mat<float>(n, n, actual_rank, A, 271);
        CHECK_F(A.move(hamr::buffer_allocator::cuda) == 0);
        A.synchronize();
        int target_rank = actual_rank + 5;
        long t = test_rsvd<float>(n, n, actual_rank, actual_rank + 5, 271, A, 2);
        file << "square"
             << ","
             << "float"
             << "," << n << "," << n << "," << target_rank << "," << 2 << "," << t << std::endl;
        t = test_rsvd<float>(n, n, actual_rank, actual_rank + 5, 271, A, 5);
        file << "square"
             << ","
             << "float"
             << "," << n << "," << n << "," << target_rank << "," << 5 << "," << t << std::endl;
    }
    file.close();
}
