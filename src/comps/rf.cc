#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <hamr_buffer.h>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::rf {

template <typename T>
void RF<T>::rf1(
    int64_t m,
    int64_t n,
    const hamr::buffer<T>& A,
    int64_t k,
    hamr::buffer<T>& Q
){
    using namespace blas;
    using namespace lapack;
    
    hamr::buffer<T> Omega(A.get_allocator(), n * k, 0.0);
    T* Omega_dat = Omega.data();
    T* Q_dat = Q.data();

    this->RS_Obj.call(m, n, A, k, Omega);

    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega_dat, n, 0.0, Q_dat, m);

    if(this->cond_check)
        // Writes into this->cond_nums
        cond_num_check<T>(m, k, Q, this->cond_nums, this->verbosity);
    
    this->Orth_Obj.call(m, k, Q);
}

template void RF<float>::rf1(int64_t m, int64_t n, const hamr::buffer<float>& A, int64_t k, hamr::buffer<float>& Q);
template void RF<double>::rf1(int64_t m, int64_t n, const hamr::buffer<double>& A, int64_t k, hamr::buffer<double>& Q);
} // end namespace RandLAPACK::comps::rf
