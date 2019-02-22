#include "Diffusion.cuh"

#include "solver/LinearSystemDevice.h"
#include <memory>

using namespace fast;

__device__ __constant__ DiffusionSysParams const_diffusion_params;
//DiffusionSysParams const_diffusion_params_cpu;

bool fast::DiffusionCommitSysParams(const DiffusionSysParams & sysparams) {
	cudaError_t res = cudaMemcpyToSymbol(
		const_diffusion_params,
		&sysparams,
		sizeof(DiffusionSysParams),
		0,
		cudaMemcpyHostToDevice
	);

	//const_diffusion_params_cpu = sysparams;

	return res == cudaSuccess;
}


////////////////////////////////


template <typename T>
__global__ void ___linearSysKernel(
	CUDA_Volume domain,
	LinearSys_StencilDevicePtr<T> A,
	T * x,
	T * b
) {

	VOLUME_VOX_GUARD(domain.res);
	size_t i = _linearIndex(domain.res, vox);

	T bval = 0.0;
	T xval = 0.0;
	Stencil_7<T> stencil;

	//(c > 0) ? T(value_one) : T(value_zero)

	Stencil_7<T> diffusivity;
	diffusivity.v[DIR_NONE] = read<T>(domain.surf, vox);
	diffusivity.v[X_NEG] = (read<T>(domain.surf, clampedVox(domain.res, vox, X_NEG)) + diffusivity.v[DIR_NONE]) * T(0.5);
	diffusivity.v[Y_NEG] = (read<T>(domain.surf, clampedVox(domain.res, vox, Y_NEG)) + diffusivity.v[DIR_NONE]) * T(0.5);
	diffusivity.v[Z_NEG] = (read<T>(domain.surf, clampedVox(domain.res, vox, Z_NEG)) + diffusivity.v[DIR_NONE]) * T(0.5);
	diffusivity.v[X_POS] = (read<T>(domain.surf, clampedVox(domain.res, vox, X_POS)) + diffusivity.v[DIR_NONE]) * T(0.5);
	diffusivity.v[Y_POS] = (read<T>(domain.surf, clampedVox(domain.res, vox, Y_POS)) + diffusivity.v[DIR_NONE]) * T(0.5);
	diffusivity.v[Z_POS] = (read<T>(domain.surf, clampedVox(domain.res, vox, Z_POS)) + diffusivity.v[DIR_NONE]) * T(0.5);

	getSystemTopKernel<T>(
		const_diffusion_params,
		diffusivity, 
		ivec3(vox.x, vox.y, vox.z), 
		ivec3(domain.res.x, domain.res.y, domain.res.z),
		&stencil, 
		&bval, 
		&xval
	);

#pragma unroll
	for (int k = 0; k <= DIR_NONE; k++) {
		A.dir[k][i] = stencil.v[k];
	}
	b[i] = bval;
	x[i] = xval;
}


template <typename T>
bool __DiffusionGenerateSystem_Impl(const CUDA_Volume & domain, LinearSystemDevice<T> * lsptr) {
	//auto lsptr = std::make_unique<LinearSystemDevice<T>>();

	auto & ls = *lsptr;
	ls.res = ivec3(domain.res.x, domain.res.y, domain.res.z);
	ls.NNZ = domain.res.x*domain.res.y*domain.res.z;

	for (auto i = 0; i < 7; i++) {
		cudaDeviceSynchronize();
		ls.A.dir[i].resize(ls.NNZ);
		cudaMemset(THRUST_PTR(ls.A.dir[i]), 0, sizeof(T) * ls.NNZ);
	}
	cudaDeviceSynchronize();
	ls.x.resize(ls.NNZ);
	cudaDeviceSynchronize();
	ls.b.resize(ls.NNZ);

	{
		BLOCKS3D(8, ls.res);
		LAUNCH(___linearSysKernel<T>, numBlocks, block,
			domain,
			ls.A.getPtr(),
			THRUST_PTR(ls.x),
			THRUST_PTR(ls.b)
		);
	}

	return true;
}

bool fast::DiffusionGenerateSystem(const CUDA_Volume & domain, LinearSystem * linearSystemOut)
{
	if (linearSystemOut == nullptr)
		return false;

	

	if (domain.type == TYPE_DOUBLE) {
		assert(dynamic_cast<LinearSystemDevice<double>*>(linearSystemOut));

		return __DiffusionGenerateSystem_Impl<double>(
			domain, 
			static_cast<LinearSystemDevice<double>*>(linearSystemOut)
		);
	}

	if (domain.type == TYPE_FLOAT) {
		assert(dynamic_cast<LinearSystemDevice<float>*>(linearSystemOut));

		return __DiffusionGenerateSystem_Impl<float>(
			domain, 
			static_cast<LinearSystemDevice<float>*>(linearSystemOut)
		);
	}

	return false;


}