#pragma once

#include <cuda_runtime.h>
#include "cuda/CudaMath.h"
#include "cuda/VolumeTypes.cuh"


#include <distfun/distfun.hpp>

namespace fast {

	struct PrimitiveCollisionPair{		
		int indexA;
		int indexB;
		distfun::AABB bounds;		
	};


	distfun::vec4 launchElasticityKernel(
		const std::vector<PrimitiveCollisionPair> & pairs
	);

	

}