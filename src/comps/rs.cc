#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <hamr_buffer.h>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::rs {

template <typename T>
void RS<T>::rs1(
	int64_t m,
	int64_t n,
	const hamr::buffer<T>& A,
	int64_t k,
	hamr::buffer<T>& Omega,
	lapack::Queue* queue 
){
	if(queue != nullptr) LOG_F(1, "rs1 on GPU");
	else LOG_F(1, "rs1 on CPU");
	using namespace blas;
	using namespace lapack;
	// The blas ops require a blas::Queue as input.
	blas::Queue* blas_queue = queue;

	int64_t p = this->passes_over_data;
	int64_t q = this->passes_per_stab;
	LOG_F(1, "rs1: %i passes over data. Stabilization once every %i pass", p, q);
	int32_t seed = this->seed;
	int64_t p_done= 0;
	

	hamr::buffer<T> Omega_1(A.get_allocator(), m * k);

	if (p % 2 == 0) 
	{
		// RandBLAS does not support CUDA at the moment.
		CHECK_F(Omega.move(hamr::buffer_allocator::cpp) == 0);
		Omega.synchronize();
		// Fill n by k Omega
		profile_timer.start_tag("gen_rand");
		RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega.data(), seed);
		profile_timer.accumulate_tag("gen_rand");
		CHECK_F(Omega.move(A.get_allocator()) == 0);
		Omega.synchronize();
	}
	else
	{	
		CHECK_F(Omega_1.move(hamr::buffer_allocator::cpp) == 0);
		Omega_1.synchronize();
		// Fill m by k Omega_1
		profile_timer.start_tag("gen_rand");
		RandBLAS::dense_op::gen_rmat_norm<T>(m, k, Omega_1.data(), seed);
		profile_timer.accumulate_tag("gen_rand");
		CHECK_F(Omega_1.move(A.get_allocator()) == 0);
		Omega_1.synchronize();
		// multiply A' by Omega results in n by k omega
		profile_timer.start_tag("gemm");
		if(queue) {
			gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A.data(), m, Omega_1.data(), m, 0.0, Omega.data(), n, *blas_queue);
		    queue->sync();
		}
		else gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A.data(), m, Omega_1.data(), m, 0.0, Omega.data(), n);
        profile_timer.accumulate_tag("gemm");
		++ p_done;
		// If q == 1
		if (p_done % q == 0) 
		{			
			profile_timer.start_tag("stab");
			this->Stab_Obj.call(n, k, Omega, queue);
			profile_timer.accumulate_tag("stab");

		}
	}
	
	while (p - p_done > 0) 
	{
		// Omega (m, k) = A * Omega (n, k)
		profile_timer.start_tag("gemm");
		if(queue) {
			gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Omega_1.data(), m, *blas_queue);
		    queue->sync();
		}
		else gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Omega_1.data(), m);
        profile_timer.accumulate_tag("gemm");

		++ p_done;
		if (p_done % q == 0) 
		{
			profile_timer.start_tag("stab");
			this->Stab_Obj.call(m, k, Omega_1, queue);
			profile_timer.accumulate_tag("stab");
		}

		if(this->cond_check)
			cond_num_check<T>(m, k, Omega_1, this->cond_nums, this->verbosity);

		// Omega (n, k) = A^T * Omega (m, k)
		profile_timer.start_tag("gemm");
		if(queue) {
			gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A.data(), m, Omega_1.data(), m, 0.0, Omega.data(), n, *blas_queue);
			queue->sync();
		}
		else gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A.data(), m, Omega_1.data(), m, 0.0, Omega.data(), n);
        profile_timer.accumulate_tag("gemm");
		++ p_done;
		if (p_done % q == 0) 
		{
			profile_timer.start_tag("stab");
			this->Stab_Obj.call(n, k, Omega, queue);
			profile_timer.accumulate_tag("stab");

		}
		
		if(this->cond_check)
			cond_num_check<T>(n, k, Omega, this->cond_nums, this->verbosity);
	}
	// Increment seed upon termination
	this->seed += m * n;
}

template void RS<float>::rs1(int64_t m, int64_t n, const hamr::buffer<float>& A, int64_t k, hamr::buffer<float>& Omega, lapack::Queue* queue);
template void RS<double>::rs1(int64_t m, int64_t n, const hamr::buffer<double>& A, int64_t k, hamr::buffer<double>& Omega, lapack::Queue* queue);
} // end namespace RandLAPACK::comps::rs
