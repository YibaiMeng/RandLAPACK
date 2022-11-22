#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <hamr_buffer.h>

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
            hamr::buffer<T> &VT)
        {
                using namespace blas;
                using namespace lapack;

                blas::Queue *queue = new blas::Queue(0, 0); // nullptr;
                if (A.cuda_accessible())
                {
                        queue = new blas::Queue(0, 0);
                }
                int64_t k = n_oversamples + rank;
                if (k > n)
                {
                        std::cerr << "rank + oversampling parameter must be lower than height of matrix" << std::endl;
                        return;
                }
                // First, use a range_finder to compute Q
                // Q = find_range(A, n_samples, n_subspace_iters)
                // Q: (m, k)
                hamr::buffer<T> Q(A.get_allocator(), m * k);
                range_finder_obj.call(m, n, A, k, Q);
                // B = Q.T @ A
                // B: (k, n)
                hamr::buffer<T> B(A.get_allocator(), k * n);
                if (queue)
                {
                        T *Q_dat = Q.data();
                        T *A_dat = A.data();
                        T *B_dat = B.data();
                        gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, (T)1.0, Q_dat, m, A_dat, m, (T)0.0, B_dat, k, (*queue));
                        queue->join();
                }
                else
                        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, (T)1.0, Q.data(), m, A.data(), m, (T)0.0, B.data(), k);
                // U_tilde, S, Vt = np.linalg.svd(B)
                // U_tilde (k, k), S k, VT (n, n)
                hamr::buffer<T> U_tilde(hamr::buffer_allocator::cpp, k * k);
                hamr::buffer<T> VT_cpu(hamr::buffer_allocator::cpp, n * n);
                // Lapackpp has no GPU version of the svd
                // need to copy to cpu.
                auto ptr = B.get_cpu_accessible();
                if (queue)
                        queue->sync();
                B.synchronize();
                gesdd(lapack::Job::AllVec, k, n, ptr.get(), k, S.data(), U_tilde.data(), k, VT.data(), n);
                U_tilde.move(A.get_allocator());
                // U = Q @ U_tilde
                // U (m, k)
                // hamr::buffer<T> U(A.get_allocator(),  m * k);
                if (queue)
                {
                        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, Q.data(), m, U_tilde.data(), k, 0.0, U.data(), k, *queue);
                        queue->sync();
                }
                else
                        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, Q.data(), m, U_tilde.data(), k, 0.0, U.data(), k);

                // Truncate
                // U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
        }

        template void RSVD<float>::call(int64_t m, int64_t n, hamr::buffer<float> &A, int64_t, int64_t, int64_t, hamr::buffer<float> &U, hamr::buffer<float> &S, hamr::buffer<float> &VT);
        template void RSVD<double>::call(int64_t m, int64_t n, hamr::buffer<double> &A, int64_t, int64_t, int64_t, hamr::buffer<double> &U, hamr::buffer<double> &S, hamr::buffer<double> &VT);
} // end namespace orth