#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif
#include <hamr_buffer.h>

namespace RandLAPACK::comps::util {


// Behaves like A[:,column_start:column_end] in python.
// fancy syntax like [1:-1] not supported.
// Expects A to be column major
template <typename T>
void slice_columns(int64_t m, 
                int64_t n, 
                hamr::buffer<T>& src, 
                int64_t column_start, 
                int64_t column_end,
                hamr::buffer<T>& dst);


template <typename T>
void eye(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& A
);

/*
Extracts the l-portion of the GETRF result, places 1's on the main diagonal.
*/
template <typename T> 
void get_L(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& L
);

/*
Overwrites the diagonal entries of matrix S with those stored in s.
*/
template <typename T>
void diag(
        int64_t m,
        int64_t n,
        const hamr::buffer<T>& s,
        int64_t k,
        hamr::buffer<T>& S
);

/*
Displays the first k diagonal elements.
*/
template <typename T> 
void disp_diag(
        int64_t m,
        int64_t n,
        int64_t k, 
        hamr::buffer<T>& A 
);

template <typename T> 
void swap_rows(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& A,
        const hamr::buffer<int64_t>& p
);

/*
Checks if the given size is larger than available. If so, resizes the vector.
*/
template <typename T> 
T* upsize(
        int64_t target_sz,
        hamr::buffer<T>& A
);

template <typename T> 
T* row_resize(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& A,
        int64_t k
);

/*
Dimensions m and n may change if we want the diagonal matrix of rank k < min(m, n).
In that case, it would be of size k by k.
*/
template <typename T> 
void gen_mat_type(
        int64_t& m, // These may change
        int64_t& n,
        hamr::buffer<T>& A,
        int64_t k, 
        int32_t seed,
        std::tuple<int, T, bool> type
);

/*
Generates matrix with the following singular values:
sigma_i = 1 / (i + 1)^pow (first k * 0.2 sigmas = 1
Can either be a diagonal matrix, or a full one.
In later case, left and right singular vectors are randomly-generated 
and orthogonaized.
*/
template <typename T> 
void gen_poly_mat(
        int64_t& m,
        int64_t& n,
        hamr::buffer<T>& A,
        int64_t k,
        T t, // controls the decay. The higher the value, the faster the decay
        bool diagon,
        int32_t seed
);

/*
Generates matrix with the following singular values:
sigma_i = e^((i + 1) * -pow) (first k * 0.2 sigmas = 1
Can either be a diagonal matrix, or a full one.
In later case, left and right singular vectors are randomly-generated 
and orthogonaized.
*/
template <typename T> 
void gen_exp_mat(
        int64_t& m,
        int64_t& n,
        hamr::buffer<T>& A,
        int64_t k,
        T t, // controls the decay. The higher the value, the faster the decay
        bool diagon,
        int32_t seed
);

/*
Generates matrix with the following singular values:
S-SHAPED DECAY (first k * 0.2 sigmas = 1)
Can either be a diagonal matrix, or a full one.
In later case, left and right singular vectors are randomly-generated 
and orthogonaized.
*/
template <typename T> 
void gen_s_mat(
        int64_t& m,
        int64_t& n,
        hamr::buffer<T>& A,
        int64_t k,
        bool diagon,
        int32_t seed
);

/*
Generates left and right singular vectors for the three matrix types above.
*/
template <typename T> 
void gen_mat(
        int64_t m,
        int64_t n,
        hamr::buffer<T>& A,
        int64_t k,
        hamr::buffer<T>& S,
        int32_t seed
);

/*
Find the condition number of a given matrix A.
*/
template <typename T> 
void cond_num_check(
        int64_t m,
        int64_t n,
        const hamr::buffer<T>& A,
        std::vector<T>& cond_nums,
        bool verbosity
);

/*
Checks whether matrix A has orthonormal columns.
*/
template <typename T> 
bool orthogonality_check(
        int64_t m,
        int64_t n,
        int64_t k,
        const hamr::buffer<T>& A,
        hamr::buffer<T>& A_gram,
        bool verbosity
);

template <typename T> 
void print_mat(
        int m,
        int n,
        hamr::buffer<T>& A,
        bool python = false
);
} // end namespace RandLAPACK::comps::rs
