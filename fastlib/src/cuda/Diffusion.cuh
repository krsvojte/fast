/*
	Diffusion linear system specific CUDA functions
	Called from Diffusion.h/cpp
*/
#pragma once
#include "cuda/Volume.cuh"
#include "cuda/LinearSystemDevice.cuh"

#include <memory>

namespace fast {
	class LinearSystem;


	//Helper struct
	template<typename T>
	struct Stencil_7 {
		T v[7];
	};


	//Construction params
	struct DiffusionSysParams {
		double highConc;
		double lowConc;
		double concetrationBegin;
		double concetrationEnd;
		double cellDim[3];
		double faceArea[3];
		int dirPrimary;
		uint2 dirSecondary;
		Dir dir;
	};

	/*
	Commits system parameters to CUDA constant memory
	*/
	bool DiffusionCommitSysParams(const DiffusionSysParams & sysparams);


	/*
	Generates linear system in a uniform grid
	*/
	bool DiffusionGenerateSystem(
		const CUDA_Volume & domain,
		LinearSystem * linearSystemOut
	);




	// Lin.sys at top level
	template <typename T>
	__host__ __device__ void getSystemTopKernel(
		//const CUDA_Volume & domain,
		const DiffusionSysParams & params,
		const Stencil_7<T> & diffusivity,
		const ivec3 & vox,
		const ivec3 & res,
		Stencil_7<T> * out,
		T * f,
		T * x
	) {

		
		const T Dneg[3] = {
			diffusivity.v[X_NEG],
			diffusivity.v[Y_NEG],
			diffusivity.v[Z_NEG]
		};
		const T Dpos[3] = {
			diffusivity.v[X_POS],
			diffusivity.v[Y_POS],
			diffusivity.v[Z_POS]
		};



		T coeffs[7];
		bool useInMatrix[7];

		coeffs[DIR_NONE] = T(0);
		useInMatrix[DIR_NONE] = true;

		for (uint j = 0; j < DIR_NONE; j++) {
			const uint k = _getDirIndex(Dir(j));
			const int sgn = _getDirSgn(Dir(j));
			const T Dface = (sgn == -1) ? Dneg[k] : Dpos[k];

			T cellDist[3] = { T(params.cellDim[0]) , T(params.cellDim[1]), T(params.cellDim[2]) };
			useInMatrix[j] = true;

			if ((vox[k] == 0 && sgn == -1) ||
				(vox[k] == res[k] - 1 && sgn == 1)
				) {
				cellDist[k] = params.cellDim[k] * T(0.5);
				useInMatrix[j] = false;
			}

			coeffs[j] = (Dface * params.faceArea[k]) / cellDist[k];

			//Subtract from diagonal
			if (useInMatrix[j] || k == params.dirPrimary)
				coeffs[DIR_NONE] -= coeffs[j];
		}


		/*
		Calculate right hand side
		*/
		const int primaryRes = ((int*)&res)[params.dirPrimary];
		T rhs = T(0);
		if (vox[params.dirPrimary] == 0) {
			Dir dir = _getDir(params.dirPrimary, -1);
			rhs -= coeffs[dir] * params.concetrationBegin;
		}
		else if (vox[params.dirPrimary] == primaryRes - 1) {
			Dir dir = _getDir(params.dirPrimary, 1);
			rhs -= coeffs[dir] * params.concetrationEnd;
		}

		*f = rhs;

		/*
		Initial guess
		*/
		const int primaryVox = vox[params.dirPrimary];
		if (_getDirSgn(params.dir) == 1)
			*x = 1.0f - (primaryVox / T(primaryRes + 1));
		else
			*x = (primaryVox / T(primaryRes + 1));


#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for (uint j = 0; j < DIR_NONE; j++) {
			if (!useInMatrix[j])
				coeffs[j] = T(0);
		}

#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for (uint i = 0; i < 7; i++) {
			out->v[i] = coeffs[i];
		}
	}



}