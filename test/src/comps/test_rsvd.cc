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

// rand (m, k) * rand(k, n) = (m, n) with a (approx) rank of k. k < min(m, n)
template <typename T>
static void gen_rand_mat(int m, int n, int k, hamr::buffer<T> &buff)
{
}


class TestRsvd : public ::testing::Test
{
protected:
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void test_rsvd(int64_t m, int64_t n, int64_t rank, int64_t target_rank, uint32_t seed, hamr::buffer_allocator alloc, bool check_diff = false)
    {
        LOG_F(INFO, "RSVD on %i by %i matrix", m, n);
        // Subroutine parameters
        bool verbosity = true;
        bool cond_check = false;
        // bool orth_check = true;

        // Make subroutine objects
        // Stabilization Constructor - Choose default
        Stab<T> Stab(0);
        int64_t n_subspace_iters = 5, passes_per_stablizer = 5;
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
        LOG_F(INFO, "Actual rank of test input is %i. Target rank %i, number of samples %i", rank, target_rank, k);
        hamr::buffer<T> U(alloc, m * target_rank);
        hamr::buffer<T> VT(alloc, n * n);
        hamr::buffer<T> S(alloc, k);


        // The cublas implementation has a bug: if you do not init everything to a value,
        // then there might be NaNs, even if beta is 0.0 in GEMM.
        hamr::buffer<T> A(hamr::buffer_allocator::cpp, m * n, 0.0);
        hamr::buffer<T> src_1(hamr::buffer_allocator::cpp, m * rank, 0.0), src_2(hamr::buffer_allocator::cpp, rank * n, 0.0);
        RandBLAS::dense_op::gen_rmat_norm<T>(m, rank, src_1.data(), seed);
        RandBLAS::dense_op::gen_rmat_norm<T>(rank, n, src_2.data(), seed + 1);
        blas::gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, rank, 1.0, src_1.data(), m, src_2.data(), rank, 0, A.data(), m);
        A.move(alloc);
        A.synchronize();
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
        U.move(hamr::buffer_allocator::cpp);
        U.synchronize();
        VT.move(hamr::buffer_allocator::cpp);
        VT.synchronize();
        S.move(hamr::buffer_allocator::cpp);
        S.synchronize();
        A.move(hamr::buffer_allocator::cpp);
        A.synchronize();
        print_mat(m, target_rank, U);
        print_mat(n, n, VT);
        LOG_F(INFO, "Calculating exact SVD");
        hamr::buffer<T> U_gold(hamr::buffer_allocator::cpp, m * m, 0.0);
        hamr::buffer<T> S_gold(hamr::buffer_allocator::cpp, n, 0.0);
        hamr::buffer<T> VT_gold(hamr::buffer_allocator::cpp, n * n, 0.0);
        auto start_svd = high_resolution_clock::now();
        lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, m, n, A.data(), m, S_gold.data(), U_gold.data(), m, VT_gold.data(), n);
        auto stop_svd = high_resolution_clock::now();
        long dur_svd = duration_cast<microseconds>(stop_svd - start_svd).count();
        LOG_F(INFO, "SVD completed in %ld us", dur_svd);
        if(check_diff) {
            LOG_F(INFO, "Reconstructing A from top %i singular values of the exact SVD", target_rank);
            hamr::buffer<T> A_reconstruct_gold(hamr::buffer_allocator::cpp, m * n, 0.0);
            hamr::buffer<T> U_tmp_gold(hamr::buffer_allocator::cpp, m * n, 0.0);
            hamr::buffer<T> S_diag_gold(hamr::buffer_allocator::cpp, target_rank * n, 0.0);
            diag<T>(target_rank, n, S_gold, target_rank, S_diag_gold);
            // U_tmp_gold = U_gold @ S_diag_gold
            blas::gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, target_rank, 1.0, U_gold.data(), m, S_diag_gold.data(), target_rank, 0, U_tmp_gold.data(), m);
            // A_reconstruct = U_tmp @ VT
            blas::gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, n, 1.0, U_tmp_gold.data(), m, VT_gold.data(), n, 0, A_reconstruct_gold.data(), m);
            print_mat(1, target_rank, S_gold);
            print_mat(1, target_rank, S);
            LOG_F(INFO, "Reconstructing A from RSVD results");
            hamr::buffer<T> A_reconstruct(hamr::buffer_allocator::cpp, m * n, 0.0);
            hamr::buffer<T> U_tmp(hamr::buffer_allocator::cpp, m * n, 0.0);
            hamr::buffer<T> S_diag(hamr::buffer_allocator::cpp, target_rank * n, 0.0);
            diag<T>(target_rank, n, S, target_rank, S_diag);
            // U_tmp = U @ S_diag
            blas::gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, target_rank, 1.0, U.data(), m, S_diag.data(), target_rank, 0, U_tmp.data(), m);
            // A_reconstruct = U_tmp @ VT
            blas::gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, n, 1.0, U_tmp.data(), m, VT.data(), n, 0, A_reconstruct.data(), m);
            // Calculate the Frobenius norm relative to the original matrix.
            blas::axpy<T>(m * n, -1.0, A_reconstruct.data(), 1, A_reconstruct_gold.data(), 1);
            T norm_orig = lapack::lange(lapack::Norm::Fro, m, n, A.data(), m);
            T norm_diff = lapack::lange(lapack::Norm::Fro, m, n, A_reconstruct_gold.data(), m);
            LOG_F(INFO, "Frobenius norm of the difference between reconstructions: %f; Norm of input: %f", norm_diff, norm_orig);
        }
    }
};

TEST_F(TestRsvd, SimpleTest)
{
    // Testing square matrices
    std::vector<int> square_matrice_size({100, 500, 1000, 2000, 4000, 5000, 8000});
    /*
    for(int n : square_matrice_size) {
        test_rsvd<double>(n, n, (int)(n * 0.1), (int)(n * 0.1 + 5), 271, hamr::buffer_allocator::cuda, false);
    }*/
    // Testing CPU version
    for(int n : square_matrice_size) {
        //test_rsvd<float>(5, 5, 2, 1, 271, hamr::buffer_allocator::cpp, false);

        test_rsvd<double>(n, n, (int)(n * 0.1), (int)(n * 0.1 + 5), 271, hamr::buffer_allocator::cpp, true);
    }
    // float32
    // Testing square matrices
    for(int n : square_matrice_size) {
        test_rsvd<float>(3, 3, 2, 1, 271, hamr::buffer_allocator::cuda, false);
    }
    // Testing CPU version
    for(int n : square_matrice_size) {
        test_rsvd<float>(n, n, (int)(n * 0.1), (int)(n * 0.1 + 5), 271, hamr::buffer_allocator::cpp, false);
    }


}
