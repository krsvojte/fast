#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/system_error.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/error.h>
#include <thrust/unique.h>
#include <stdio.h>


#define FAST_MANAGED_ALWAYS

template<class T>
class managed_allocator : public thrust::device_malloc_allocator<T>
{
public:
	using value_type = T;

	typedef thrust::device_ptr<T>  pointer;
	inline pointer allocate(size_t n)
	{
		value_type* result = nullptr;
#ifdef FAST_MANAGED_DISABLED
		cudaError_t error = cudaMalloc(&result, n * sizeof(T));
#elseif FAST_MANAGED_ALWAYS
		cudaError_t error = cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal);

		if (error != cudaSuccess)
		{
			throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
		}
#else

		size_t free;
		size_t total;
		cudaMemGetInfo(&free, &total);		
		

		cudaError_t error = cudaErrorMemoryAllocation;

		if (free < 0.25 * total) {
			//Try normal alloc			
			error = cudaMalloc(&result, n * sizeof(T));			
		}

		if(error != cudaSuccess){
			//If that doesn't work, try managed alloc		
			error = cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal);
			
		}
		

		if (error != cudaSuccess)
		{
			fflush(NULL);
			throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
		}

#endif


		return thrust::device_pointer_cast(result);
	}

	inline void deallocate(pointer ptr, size_t)
	{
		cudaError_t error = cudaFree(thrust::raw_pointer_cast(ptr));

		if (error != cudaSuccess)
		{
			throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
		}
	}
};

template<class T>
using managed_device_vector = thrust::device_vector<T, managed_allocator<T>>;


#define MANAGED_POLICY(T) thrust::cuda::par(managed_allocator<T>())


template <typename T>
class scheduled_vector {


};
