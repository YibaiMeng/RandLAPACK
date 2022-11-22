/*
TODO #1: The QB class probably doesn't need a QB2 function that's separate from call(...). 
The same goes for RF not needing rf1 separately from call(...). I know that the Orth class has 
several functions and a call method that chooses among them. That's an okay design pattern in isolation 
but if we assume it's used in most implementations then that can lead to indirect assumptions about how 
customized user implementations might behave.
*/

#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif
#include <hamr_buffer.h>

#include "orth.hh"
#include "rf.hh"

namespace RandLAPACK::comps::qb {

#ifndef QB_CLASS
#define QB_CLASS

template <typename T>
class QBalg
{
        public:
                virtual void call(
                        int64_t m,
                        int64_t n,
                        hamr::buffer<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        hamr::buffer<T>& Q,
                        hamr::buffer<T>& B
                ) = 0;
};

template <typename T>
class QB : public QBalg<T>
{
	public:
                RandLAPACK::comps::rf::RangeFinder<T>& RF_Obj;
                RandLAPACK::comps::orth::Stabilization<T>& Orth_Obj;
                bool verbosity;
                bool orth_check;

                hamr::buffer<T> Q_gram;
                hamr::buffer<T> Q_i_gram;

                hamr::buffer<T> QtQi; 
                hamr::buffer<T> Q_i;
                hamr::buffer<T> B_i;

                // Controls QB version to be used
                int decision_QB;

                // Output
                int termination;

                /*
                This represents how much space is currently allocated for cols of Q and rows of B.
                This is <= k. We are assuming that the user may not have given "enough"
                space when allocating Q, B initially.
                */
                int64_t curr_lim;

                // By how much are we increasing the dimension when we've reached curr_lim
                int dim_growth_factor;

		// Constructor
		QB(
                        RandLAPACK::comps::rf::RangeFinder<T>& rf_obj,
                        RandLAPACK::comps::orth::Stabilization<T>& orth_obj,
                        bool verb,
                        bool orth,
                        int decision,
                        hamr::buffer_allocator default_allocator = hamr::buffer_allocator::cpp
		) : RF_Obj(rf_obj), Orth_Obj(orth_obj), Q_gram(default_allocator, 0, (T)(0)), Q_i_gram(default_allocator, 0, (T)(0)), 
                    QtQi(default_allocator, 0, (T)(0)),  Q_i(default_allocator, 0, (T)(0)),  B_i(default_allocator, 0, (T)(0)) 
                {
                        verbosity = verb;
                        orth_check = orth;
                        decision_QB = decision;
                        dim_growth_factor = 4;
		}

		void QB2(
                        int64_t m,
                        int64_t n,
                        hamr::buffer<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        hamr::buffer<T>& Q,
                        hamr::buffer<T>& B
                );

                virtual void call(
                        int64_t m,
                        int64_t n,
                        hamr::buffer<T>& A,
                        int64_t& k,
                        int64_t block_sz,
                        T tol,
                        hamr::buffer<T>& Q,
                        hamr::buffer<T>& B
                )
                {
                        switch(this->decision_QB)
                        {
                                case 0:
                                        QB2(m, n, A, k, block_sz, tol, Q, B);
                                        break;
                        }
                }
};
#endif
} // end namespace RandLAPACK::comps::qb
