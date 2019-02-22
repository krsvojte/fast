#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/Types.h>

#include <array>

namespace fast {

	class Volume;
	class Volume;
	struct VolumeCCL;
	

	enum DiffusionSolverType {
		DSOLVER_BICGSTAB,		
		DSOLVER_CG
	};

	struct TortuosityParams {
		Dir dir = X_NEG;
		vec2 coeffs = vec2(1.0, 0.001);
		double tolerance = 1e-6;		
		size_t maxIter = 10000;
		bool verbose = false;
		bool oversubscribeGPUMemory = false;
		bool useNonUniform = false;

		//Porosity
		bool porosityPrecomputed = false;
		double porosity = -1.0;
		bool onDevice = true;
		int cpuThreads = -1;		
	};

	/*
		Calculates tortuosity
		Inputs: 
			binary volume mask
			Tortuosity params detailing direction, tolerance, etc.
			Solver to use,
		Outputs:
			Returns tortuosity.
			Returns 0 on error
			If concetrationOutput is specified, 
			the concetration from diffusion equation is stored there.
		Notes:
			Calculates porosity if not provided

	*/
	template <typename T> 
	FAST_EXPORT T getTortuosity(
		const Volume & mask,  
		const TortuosityParams & params,
		DiffusionSolverType solverType = DSOLVER_BICGSTAB,
		Volume * concetrationOutput = nullptr, //Optional output of diffusion
		size_t * iterationsOut = nullptr //optional iterations output
	);

	template <typename T>
	FAST_EXPORT T getPorosity(const Volume & mask);


	template <typename T>
	FAST_EXPORT T getReactiveAreaDensity(
		const Volume & mask, 
		ivec3 res, float isovalue//, float smooth,
		//uint * vboOut = nullptr, //If not null, a triangle mesh will be generated and saved to vbo
		//size_t * NvertsOut = nullptr
	);

	template <typename T>
	FAST_EXPORT std::array<T, 6> getReactiveAreaDensityTensor(
		const VolumeCCL & ccl,
		ivec3 res = ivec3(0,0,0),
		float isovalue = 0.5f
	);

	template <typename T>
	FAST_EXPORT T getShapeFactor(		
		T averageParticleArea,
		T averageParticleVolume
	);
	
	
	

}