

#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif
#include <hamr_buffer.h>

namespace RandLAPACK::comps::rsvd
{

        template <typename T>
        class RSVDalg
        {
        public:
                virtual void call(
                    int64_t m,                // Width of matrix.
                    int64_t n,                // Height of input matrix.
                    hamr::buffer<T> &A,       // Input matrix
                    int64_t rank,             // Target rank apporximation.
                    int64_t n_oversamples,    // Oversampling parameter
                    hamr::buffer<T> &U,
                    hamr::buffer<T> &S,
                    hamr::buffer<T> &VT,
                    lapack::Queue* q  = nullptr // Execution context for device.
                    ) = 0;
        };

        /// The sketching implementation with fixed rank objective, as shown in Halko 2011.
        template <typename T>
        class RSVD : public RSVDalg<T>
        {
        public:
                rs::RowSketcher<T> &row_sketcher_obj;
                orth::Stabilization<T> &stabilization_obj;
                rf::RangeFinder<T> &range_finder_obj;
                // Constructor
                RSVD(
                    rs::RowSketcher<T> &row_sketcher_obj,
                    orth::Stabilization<T> &stabilization_obj,
                    rf::RangeFinder<T> &range_finder_obj) : row_sketcher_obj(row_sketcher_obj), stabilization_obj(stabilization_obj), range_finder_obj(range_finder_obj)
                {
                }

                virtual void call(
                    int64_t m,                // Width of matrix.
                    int64_t n,                // Height of input matrix.
                    hamr::buffer<T> &A,       // Input matrix
                    int64_t rank,             // Target rank apporximation.
                    int64_t n_oversamples,    // Oversampling parameter
                    hamr::buffer<T> &U,
                    hamr::buffer<T> &S,
                    hamr::buffer<T> &VT, 
                    lapack::Queue* q  = nullptr);
                
        };
} // end namespace RandLAPACK::comps::rsvd
