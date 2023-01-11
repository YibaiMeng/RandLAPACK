/*
TODO #1: only store the upper triangle of the gram matrix in gram_vec,
so that it can be of size k*(k+1)/2 instead of k*k.
*/

#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <hamr_buffer.h>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::orth {

// Perfoms a Cholesky QR factorization
template <typename T> 
int Orth<T>::CholQR(
        int64_t m,
        int64_t k,
        hamr::buffer<T>& Q
){
        using namespace blas;
        using namespace lapack;
        hamr::buffer<T> Q_gram(Q.get_allocator(), k * (k + 1) / 2);
        T* Q_gram_dat = Q_gram.data();
        T* Q_dat = Q.data();
        profile_timer.start_tag("sfrk");
        // Find normal equation Q'Q - Just the upper triangular portion        
        sfrk(Op::NoTrans, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, Q_gram_dat);
        // Positive definite cholesky factorization
        profile_timer.accumulate_tag("sfrk");
        profile_timer.start_tag("pftrf");
        if (pftrf(Op::NoTrans, Uplo::Upper, k, Q_gram_dat))
        {
                profile_timer.accumulate_tag("pftrf");
                this->chol_fail = true; // scheme failure 
                return 1;
        }
        profile_timer.accumulate_tag("pftrf");
        profile_timer.start_tag("tfsm");
        tfsm(Op::NoTrans, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, Q_gram_dat, Q_dat, m);
        profile_timer.accumulate_tag("tfsm");
       return 0;
}

template <typename T> 
int Stab<T>::PLU(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& A
){
        using namespace lapack;

        // Not using utility bc vector of int
        hamr::buffer<int64_t> ipiv(A.get_allocator(), n);
        profile_timer.start_tag("geqrf");
        if(getrf(m, n, A.data(), m, ipiv.data())) {
        profile_timer.accumulate_tag("geqrf");
            return 1; // failure condition
        }
        profile_timer.accumulate_tag("geqrf");

        get_L<T>(m, n, A);
        swap_rows<T>(m, n, A, ipiv);

        return 0;
}

template <typename T> 
int Orth<T>::HQR(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& A
){
        // Done via regular LAPACK's QR
        // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
        // tau needs to be a vector of all 2's by default
        using namespace lapack;
        hamr::buffer<T> tau(A.get_allocator(), n);

        T* A_dat = A.data();
	T* tau_dat = tau.data();
        profile_timer.start_tag("geqrf");
        if(geqrf(m, n, A_dat, m, tau_dat)) {
            profile_timer.accumulate_tag("geqrf");
            return 1; // Failure condition
        }
        profile_timer.accumulate_tag("geqrf");
        profile_timer.start_tag("ungqr");
        ungqr(m, n, n, A_dat, m, tau_dat);
        profile_timer.accumulate_tag("ungqr");
        return 0;
}

// GEQR lacks "unpacking" of Q
template <typename T> 
int Orth<T>::GEQR(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& A
){
        using namespace lapack;

        hamr::buffer<T> tvec(A.get_allocator(), 5); // 5 is because 

        T* A_dat = A.data();
        
        geqr(m, n, A_dat, m, tvec.data(), -1);
        auto ptr = tvec.get_cpu_accessible();
        int64_t tsize = (int64_t) ptr.get()[0]; 
        tvec.resize(tsize);
        profile_timer.start_tag("geqr");
        if(geqr(m, n, A_dat, m, tvec.data(), tsize)) {
            profile_timer.accumulate_tag("geqr");
            return 1;
        }            
        profile_timer.accumulate_tag("geqr");

        return 0;
}

template int Orth<float>::CholQR(int64_t m, int64_t k, hamr::buffer<float>& Q);
template int Orth<double>::CholQR(int64_t m, int64_t k, hamr::buffer<double>& Q);

template int Stab<float>::PLU(int64_t m, int64_t n, hamr::buffer<float>& A);
template int Stab<double>::PLU(int64_t m, int64_t n, hamr::buffer<double>& A);

template int Orth<float>::HQR(int64_t m, int64_t n, hamr::buffer<float>& A);
template int Orth<double>::HQR(int64_t m, int64_t n, hamr::buffer<double>& A);

template int Orth<float>::GEQR(int64_t m, int64_t n, hamr::buffer<float>& A);
template int Orth<double>::GEQR(int64_t m, int64_t n, hamr::buffer<double>& A); 
} // end namespace orth