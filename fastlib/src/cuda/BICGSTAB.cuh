#pragma once
#include "LinearSys.cuh"
#include "cuda/ManagedAllocator.cuh"

#include "cuda/LinearSystemDevice.cuh"
#include <thrust/fill.h>

//Holds vectors needed for BICGstab iterations
template <typename T>
struct BICGState {
	managed_device_vector<T> r0, r, p, v, h, s, t, y, z;
	bool resize(size_t N);
	
	static size_t requiredBytesize(size_t nnz) {
		return sizeof(T) * nnz * 9;
	}
};



namespace fast {
	template <typename T>
	bool BICG_Solve(
		LinearSystemDevice<T> & sys,
		BICGState<T> & state,
		T tolerance, size_t maxIter,
		T *outError, size_t * outIterations,
		bool verbose = false
	);


}


