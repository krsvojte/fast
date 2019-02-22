/*
	CUDA Functions related to general linear system on device.
	See LinearSystemDevice.h
*/
#pragma once
#include "Volume.cuh"

#include "cuda/ManagedAllocator.cuh"
#include "solver/LinearSystemDevice.h"
#include <thrust/inner_product.h>

namespace fast {
	

	/*
	Generates domain from mask and two double values
	*/
	void GenerateDomain(
		const CUDA_Volume & binaryMask,
		double value_zero,
		double value_one,
		CUDA_Volume & output
	);


#ifdef __CUDACC__

	template <typename T>
	__global__ void ___residualKernel(ivec3 res,
		const LinearSys_StencilDevicePtr<T> A,
		const T * x,
		const T * b,
		T * r
	) {
		VOLUME_IVOX_GUARD(res);
		size_t I = _linearIndex(res, ivox);

		T sum = T(0);

		const int3 stride = { 1, int(res.x), int(res.x*res.y) };
		sum += (A.dir[X_POS][I] != T(0)) ? x[I + stride.x] * A.dir[X_POS][I] : T(0);
		sum += (A.dir[X_NEG][I] != T(0)) ? x[I - stride.x] * A.dir[X_NEG][I] : T(0);
		sum += (A.dir[Y_POS][I] != T(0)) ? x[I + stride.y] * A.dir[Y_POS][I] : T(0);
		sum += (A.dir[Y_NEG][I] != T(0)) ? x[I - stride.y] * A.dir[Y_NEG][I] : T(0);
		sum += (A.dir[Z_POS][I] != T(0)) ? x[I + stride.z] * A.dir[Z_POS][I] : T(0);
		sum += (A.dir[Z_NEG][I] != T(0)) ? x[I - stride.z] * A.dir[Z_NEG][I] : T(0);
		sum += (A.dir[DIR_NONE][I] != T(0)) ? x[I] * A.dir[DIR_NONE][I] : T(0);
		r[I] = b[I] - sum;
	}

	template <typename T>
	void LinearSys_Residual(LinearSystemDevice<T> & sys, managed_device_vector<T> & r)
	{
		BLOCKS3D(8, sys.res)
			___residualKernel<T> << <numBlocks, block >> > (
				sys.res, sys.A.getPtr(), THRUST_PTR(sys.x), THRUST_PTR(sys.b), THRUST_PTR(r)
				);
	}

	template <typename Tvec>
	typename Tvec::value_type dotProduct(Tvec & a, Tvec & b) {
		cudaDeviceSynchronize();
		return thrust::inner_product(a.begin(), a.end(), b.begin(), (typename Tvec::value_type)(0));
	}

	template <typename T>
	void __global__ __matrixVecProductKernel(ivec3 res, const LinearSys_StencilDevicePtr<T> A, const T * x, T *b) {
		VOLUME_IVOX_GUARD(res);
		size_t I = _linearIndex(res, ivox);
		T sum = T(0);
		const ivec3 stride = { 1, int(res.x), int(res.x*res.y) };
		sum += (A.dir[X_POS][I] != T(0)) ? x[I + stride.x] * A.dir[X_POS][I] : T(0);
		sum += (A.dir[X_NEG][I] != T(0)) ? x[I - stride.x] * A.dir[X_NEG][I] : T(0);
		sum += (A.dir[Y_POS][I] != T(0)) ? x[I + stride.y] * A.dir[Y_POS][I] : T(0);
		sum += (A.dir[Y_NEG][I] != T(0)) ? x[I - stride.y] * A.dir[Y_NEG][I] : T(0);
		sum += (A.dir[Z_POS][I] != T(0)) ? x[I + stride.z] * A.dir[Z_POS][I] : T(0);
		sum += (A.dir[Z_NEG][I] != T(0)) ? x[I - stride.z] * A.dir[Z_NEG][I] : T(0);
		sum += (A.dir[DIR_NONE][I] != T(0)) ? x[I] * A.dir[DIR_NONE][I] : T(0);
		b[I] = sum;
	}



	template <typename T>
	struct square
	{
		__host__ __device__
			T operator()(const T& x) const {
			return x * x;
		}
	};

	template <typename Tvec>
	typename Tvec::value_type squareNorm(Tvec & vec) {

		cudaDeviceSynchronize();
		
		square<typename Tvec::value_type>        unary_op;
		thrust::plus<typename Tvec::value_type> binary_op;

		managed_allocator<typename Tvec::value_type> allocator;

		typename Tvec::value_type norm = thrust::transform_reduce(
			thrust::cuda::par(allocator),
			vec.begin(),
			vec.end(),
			unary_op,
			typename Tvec::value_type(0),
			binary_op
		);
		return norm;
	}



	template <typename T>
	void __global__ __aPlusBetaBGammaPlusCKernel(uint NNZ, T * A, const T* B, const T*C, T beta, T gamma) {
		INDEX_GRID2D_GUARD(NNZ);
		A[i] = gamma * (A[i] + beta* B[i]) + C[i];
	}



	template <typename T>
	void __global__ __aPlusBetaBKernel(size_t NNZ, const T*A, const T*B, T*C, T beta) {
		INDEX_GRID2D_GUARD(NNZ);
		C[i] = A[i] + beta * B[i];
	}

	template <typename T>
	void __global__ __aAddBetaBKernel(size_t NNZ, T*A, const T*B, T beta) {
		INDEX_GRID2D_GUARD(NNZ);
		A[i] += beta * B[i];
	}

	template <typename T>
	void __global__ __aAddAlphaBKernel(size_t NNZ, T*A, const T*B, T alpha) {
		INDEX_GRID2D_GUARD(NNZ);
		A[i] = A[i] * alpha + B[i];
	}

	template <typename T>
	void __global__ __ABC_BetaGammaKernel(size_t NNZ, T*A, const T*B, const T*C, T beta, T gamma) {
		INDEX_GRID2D_GUARD(NNZ);
		A[i] += beta * B[i] + gamma * C[i];
	}

	template <typename T>
	void __global__ __mulByInverseKernel(uint NNZ, const T * A, const T * b, T *res) {
		INDEX_GRID2D_GUARD(NNZ);
		res[i] = b[i] * T(1.0) / A[i];
	}

	template<typename T, typename K>
	__global__ void vectorToVolume(CUDA_Volume volume, const T * vec, const uint * origToIndex = nullptr) {
		VOLUME_IVOX_GUARD(volume.res);
		size_t I = _linearIndex(volume.res, ivox);

		T value = 0;

		if (origToIndex) {
			uint actualIndex = origToIndex[I];
			if (actualIndex != uint(-1))
				value = vec[actualIndex];
		}
		else {
			value = vec[I];
		}

		write<K>(volume.surf, ivox, K(value));

	}
#endif

}