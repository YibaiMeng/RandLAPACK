#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <hamr_buffer.h>
#include <loguru.hpp>
using namespace RandLAPACK::comps::util;


namespace RandLAPACK::comps::rsvd
{

        // Preforms RSVD as shown in Halko
        template <typename T>
        void RSVD<T>::call(
            int64_t m,                // Width of matrix.
            int64_t n,                // Height of input matrix.
            hamr::buffer<T> &A,       // Input matrix
            int64_t rank,             // Target rank apporximation.
            int64_t n_oversamples,    // Oversampling parameter
            int64_t n_subspace_iters, // Number of power iterations.
            hamr::buffer<T> &U,
            hamr::buffer<T> &S,
            hamr::buffer<T> &VT, 
            lapack::Queue* queue // Needed for GPU
        ) {
                using namespace blas;
                using namespace lapack;

                blas::Queue* blas_queue = nullptr;
                auto A_alloc = A.get_allocator();
                if (A.cuda_accessible())
                {  
                        CHECK_F(queue != nullptr, "Must have execution context (lapack::Queue* queue) to run code on GPU.");
                        // We need a blas queue
                        blas_queue = queue;
                        LOG_F(INFO, "Running on GPU.");
                }
                int64_t k = n_oversamples + rank;
                if (k > n)
                {
                        LOG_F(ERROR, "rank + oversampling parameter must be lower than height of matrix");
                        exit(-1);
                }
                // First, use a range_finder to compute Q
                // Q = find_range(A, n_samples, n_subspace_iters)
                // Q: (m, k)
                hamr::buffer<T> Q(A_alloc, m * k, 0.0);
                Q.synchronize();
                LOG_F(INFO, "Rangefinder started");
                range_finder_obj.call(m, n, A, k, Q, queue);
                LOG_F(INFO, "Rangefinder finished");

                // B = Q.T @ A
                // B: (k, n)
                hamr::buffer<T> B(A.get_allocator(), k * n, 0.0);

                if (queue)
                {
                        T *Q_dat = Q.data();
                        T *A_dat = A.data();
                        T *B_dat = B.data();
                        B.synchronize();
                        LOG_F(INFO, "B = Q.T @ A started on GPU");
                        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, (T)1.0, Q_dat, m, A_dat, m, (T)0.0, B_dat, k, (*blas_queue));
                }
                else {
                        LOG_F(INFO, "B = Q.T @ A started on CPU");
                        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, (T)1.0, Q.data(), m, A.data(), m, (T)0.0, B.data(), k);
                        LOG_F(INFO, "B = Q.T @ A completed on CPU");
                }
                // U_tilde, S, Vt = np.linalg.svd(B)
                // U_tilde (k, k), S (k,), VT (n, n)
                LOG_F(INFO, "Allocating U_tilde and VT_cpu on CPU");
                hamr::buffer<T> U_tilde(hamr::buffer_allocator::cpp, k * k);
                hamr::buffer<T> VT_cpu(hamr::buffer_allocator::cpp, n * n);
                // Lapackpp has no GPU version of the svd
                // need to copy to cpu.
                if (queue) {
                   queue->sync();
                   LOG_F(INFO, "B = Q.T @ A completed on GPU");
                }
                auto ptr = B.get_cpu_accessible();
                CHECK_F(S.move(hamr::buffer_allocator::cpp) == 0);
                S.synchronize();
                LOG_F(INFO, "Starting U_tilde, S, VT_cpu = gesvd(B)");
                gesvd(lapack::Job::AllVec, lapack::Job::AllVec, k, n, ptr.get(), k, S.data(), U_tilde.data(), k, VT_cpu.data(), n);
                LOG_F(INFO, "Finished U_tilde, S, VT_cpu = gesvd(B)");
                CHECK_F(U_tilde.move(A.get_allocator()) == 0);
                // TODO: test out HAMR's sync behavior regarding memory and computation?
                U_tilde.synchronize();
                // U = Q (m, k) @ U_tilde (k, k)
                // U (m, k)
                // However, only the first "rank" columns of U is needed, so U is truncated to (m, rank).
                // U (m, rank) = Q(m, k) @ U_tilde (k, rank)
                if (queue)
                {
                        LOG_F(INFO, "Starting U = Q @ U_tilde on GPU");
                        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, rank, k, 1.0, Q.data(), m, U_tilde.data(), k, 0.0, U.data(), m, *blas_queue);
                        queue->sync();
                        LOG_F(INFO, "Finishing U = Q @ U_tilde on GPU");
                }
                else {
                        LOG_F(INFO, "Starting U = Q @ U_tilde on CPU");
                        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, rank, k, 1.0, Q.data(), m, U_tilde.data(), k, 0.0, U.data(), m);
                        LOG_F(INFO, "Finishing U = Q @ U_tilde on CPU");
                }
                // Copying VT_cpu to VT
                LOG_F(INFO, "Copying VT_cpu to VT");
                VT = VT_cpu;
                LOG_F(INFO, "Copied VT_cpu to VT");
        }

        template void RSVD<float>::call(int64_t m, int64_t n, hamr::buffer<float> &A, int64_t, int64_t, int64_t, hamr::buffer<float> &U, hamr::buffer<float> &S, hamr::buffer<float> &VT, lapack::Queue* q);
        template void RSVD<double>::call(int64_t m, int64_t n, hamr::buffer<double> &A, int64_t, int64_t, int64_t, hamr::buffer<double> &U, hamr::buffer<double> &S, hamr::buffer<double> &VT, lapack::Queue* q);
} // end namespace rsvd