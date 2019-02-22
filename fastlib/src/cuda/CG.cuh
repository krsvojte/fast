#pragma once
#include "cuda/LinearSys.cuh"
#include "cuda/ManagedAllocator.cuh"


#include "cuda/LinearSystemDevice.cuh"

template <typename T>
struct CGState {
	//managed_device_vector<T> r0, r, p, v, h, s, t, y, z;// _ainvert;

	managed_device_vector<T> r, p, ap, z;
	bool resize(size_t N, bool precondition);

	static size_t requiredBytesize(size_t nnz, bool precondition) {
		return sizeof(T) * nnz * (precondition ? 4 : 3);
	}
};





/*
template <typename T>
void CG_Output(
	const LinearSys<T> & sys,
	CUDA_Volume & out
);
*/


namespace fast{

	template <typename T>
	bool CG_Solve(
		LinearSystemDevice<T> & sys,
		CGState<T> & state,
		T tolerance, size_t maxIter,
		T *outError, size_t * outIterations,
		bool verbose = false
	);
}