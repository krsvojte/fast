#include "solver/Diffusion.h"

#include "solver/LinearSystemDevice.h"
#include "solver/LinearSystemHost.h"

#include "cuda/Diffusion.cuh"
#include "cuda/LinearSystemDevice.cuh"

#include "volume/Volume.h"

#include <omp.h>

using namespace fast;


DiffusionSysParams setupSysParams(const DiffusionPrepareParams & params) {
	DiffusionSysParams sysp;
	sysp.highConc = double(1.0);
	sysp.lowConc = double(0.0);
	sysp.concetrationBegin = (getDirSgn(params.dir) == 1) ? sysp.highConc : sysp.lowConc;
	sysp.concetrationEnd = (getDirSgn(params.dir) == 1) ? sysp.lowConc : sysp.highConc;

	sysp.cellDim[0] = params.cellDim.x;
	sysp.cellDim[1] = params.cellDim.y;
	sysp.cellDim[2] = params.cellDim.z;
	sysp.faceArea[0] = sysp.cellDim[1] * sysp.cellDim[2];
	sysp.faceArea[1] = sysp.cellDim[0] * sysp.cellDim[2];
	sysp.faceArea[2] = sysp.cellDim[0] * sysp.cellDim[1];

	sysp.dirPrimary = getDirIndex(params.dir);
	sysp.dirSecondary = make_uint2((sysp.dirPrimary + 1) % 3, (sysp.dirPrimary + 2) % 3);

	sysp.dir = params.dir;
	return sysp;
}

size_t getColumn(size_t row, const ivec3 & stride, Dir d) {
	switch (d) {
		case DIR_NONE: return row;
		case X_POS: return row + stride.x;
		case X_NEG: return row - stride.x;
		case Y_POS: return row + stride.y;
		case Y_NEG: return row - stride.y;
		case Z_POS: return row + stride.z;
		case Z_NEG: return row - stride.z;		
	}
	assert(false);
	return 0;
}

size_t clampedIndex(ivec3 pos, const ivec3 & res, Dir d) {
	const int k = getDirIndex(d);
	int sgn = getDirSgn(d); 

	const int newVox = pos[k] + sgn;// _at<int>(vox, k) + sgn;
	const int & resK = res[k];// _at<int>(res, k);
	if (newVox >= 0 && newVox < resK) {
		pos[k] = newVox;
		//_at<int>(vox, k) = uint(newVox);
	}
	return linearIndex(res,pos);
}

template <typename T>
std::unique_ptr<LinearSystemHost<T>> _genDiffuseHostImpl(
	const DiffusionPrepareParams & params,
	ivec3 res,
	const uchar * mask,
	int numThreads
){

	const auto dim = res;
	const size_t N = dim.x * dim.y * dim.z;

	auto sysParam = setupSysParams(params);
	auto sys = std::make_unique<LinearSystemHost<T>>();


	sys->A.resize(N, N);
	sys->A.reserve(Eigen::VectorXi::Constant(N, 7));

	sys->x.resize(N);
	sys->b.resize(N);

	ivec3 stride = { 1, dim.x, dim.x*dim.y };

	
	const Dir columnOrder[7] = {
		Z_NEG,Y_NEG,X_NEG,DIR_NONE,X_POS,Y_POS,Z_POS
	};


	//Converts to float/double using params d0 and d1
	auto dfun = [&](uchar a) -> T{
		return (a > 0) ? T(params.d1) : T(params.d0);
	};


	
	if (numThreads < 0) {
		numThreads = omp_get_num_procs();		
	}
	omp_set_num_threads(numThreads);

	#pragma omp parallel for
	for (auto z = 0; z < dim.z; z++) {
		for (auto y = 0; y < dim.y; y++) {
			for (auto x = 0; x < dim.x; x++) {
				const ivec3 ipos = { x,y,z };
				const auto i = linearIndex(dim, ipos);
				
#ifdef _DEBUG
				int testThreadNum;  
				testThreadNum = omp_get_num_threads();
#endif
				//params.d0
				
				Stencil_7<T> diffusivity;
				diffusivity.v[DIR_NONE] = dfun(mask[i]);
				diffusivity.v[X_NEG] = (dfun(mask[clampedIndex(ipos, res, X_NEG)]) + diffusivity.v[DIR_NONE]) * T(0.5);
				diffusivity.v[Y_NEG] = (dfun(mask[clampedIndex(ipos, res, Y_NEG)]) + diffusivity.v[DIR_NONE]) * T(0.5);
				diffusivity.v[Z_NEG] = (dfun(mask[clampedIndex(ipos, res, Z_NEG)]) + diffusivity.v[DIR_NONE]) * T(0.5);
				diffusivity.v[X_POS] = (dfun(mask[clampedIndex(ipos, res, X_POS)]) + diffusivity.v[DIR_NONE]) * T(0.5);
				diffusivity.v[Y_POS] = (dfun(mask[clampedIndex(ipos, res, Y_POS)]) + diffusivity.v[DIR_NONE]) * T(0.5);
				diffusivity.v[Z_POS] = (dfun(mask[clampedIndex(ipos, res, Z_POS)]) + diffusivity.v[DIR_NONE]) * T(0.5);

				
				Stencil_7<T> out;
				T rhs;
				T xguess;
				
				getSystemTopKernel<T>(sysParam, diffusivity, ipos, res, &out, &rhs, &xguess);

				for (auto k = 0; k < 7; k++) {
					Dir d = columnOrder[k];
					if (out.v[int(d)] != T(0)) {							
						sys->A.insert(i, getColumn(i,stride,d)) = out.v[int(d)];
					}					
				}			

				sys->b[i] = rhs;
				sys->x[i] = xguess;


			}
		}
	}




	return sys;
}

FAST_EXPORT std::unique_ptr<LinearSystem> 
fast::GenerateDiffusionLinearSystemHost(
	const DiffusionPrepareParams & params, 
	ivec3 res, 
	const uchar * mask,
	int numThreads
)
{
	if (params.type == TYPE_DOUBLE) {
		return _genDiffuseHostImpl<double>(params, res, mask, numThreads);
	}

	if (params.type == TYPE_FLOAT) {
		return _genDiffuseHostImpl<float>(params, res, mask, numThreads);
	}

	return nullptr;
}




FAST_EXPORT std::unique_ptr<LinearSystem> 
fast::GenerateDiffusionLinearSystemDevice(
	const DiffusionPrepareParams & params, 
	CUDA_Volume * mask
)
{
	const auto dim = mask->res;
	const size_t N = dim.x * dim.y * dim.z;

	Volume v({ dim.x,dim.y,dim.z }, params.type );
	CUDA_Volume domain = *v.getCUDAVolume();

	/*
	Generate domain of type T
	*/
	{
		GenerateDomain(*mask, params.d0, params.d1, domain);
		if (params.verbose)
			std::cout << "Gen domain" << std::endl;
	}

	/*
	Commit system creation params to constant memory
	*/
	{
		if (!DiffusionCommitSysParams(setupSysParams(params))) {
			assert(false);
			return nullptr;
		}
		if (params.verbose)
			std::cout << "Commit sys params" << std::endl;
	}


	/*
	Return generated linear system pointer
	*/
	std::unique_ptr<LinearSystem> linsys;
	if (domain.type == TYPE_DOUBLE) {
		linsys = std::make_unique<LinearSystemDevice<double>>();
	}
	else if (domain.type == TYPE_FLOAT) {
		linsys = std::make_unique<LinearSystemDevice<float>>();
	}
	else {
		return nullptr;
	}
	
	if (DiffusionGenerateSystem(domain, linsys.get())) {
		return linsys;
	}

	return nullptr;

	
}

