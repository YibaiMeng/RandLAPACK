#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <hamr_buffer.h>

#include <loguru.hpp>
using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::rf {

template <typename T>
void RF<T>::rf1(
    int64_t m,
    int64_t n,
    const hamr::buffer<T>& A,
    int64_t k,
    hamr::buffer<T>& Q,
    lapack::Queue* queue
){
    using namespace blas;
    using namespace lapack;
    LOG_F(INFO, "Allocating Omega");
    hamr::buffer<T> Omega(A.get_allocator(), n * k, 0.0);
    T* Omega_dat = Omega.data();
    T* Q_dat = Q.data();
    LOG_F(INFO, "Starting row sketcher");
    this->RS_Obj.call(m, n, A, k, Omega, queue);
    LOG_F(INFO, "Finishing row sketcher");
    LOG_F(INFO, "Starting A @ Omega");

    // Q = orth(A * Omega)
    if(queue) gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega_dat, n, 0.0, Q_dat, m, *queue);
    else gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega_dat, n, 0.0, Q_dat, m);
    LOG_F(INFO, "Finishing A @ Omega");

    if(this->cond_check)
        // Writes into this->cond_nums
        cond_num_check<T>(m, k, Q, this->cond_nums, this->verbosity);
    LOG_F(INFO, "Starting orth(A @ Omega)");
    this->Orth_Obj.call(m, k, Q, queue);
    LOG_F(INFO, "Finishing orth(A @ Omega)");

}

template void RF<float>::rf1(int64_t m, int64_t n, const hamr::buffer<float>& A, int64_t k, hamr::buffer<float>& Q, lapack::Queue* queue);
template void RF<double>::rf1(int64_t m, int64_t n, const hamr::buffer<double>& A, int64_t k, hamr::buffer<double>& Q, lapack::Queue* queue);
} // end namespace RandLAPACK::comps::rf
