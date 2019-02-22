#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/Types.h>
#include <fastlib/solver/LinearSystem.h>


#include <memory>


struct CUDA_Volume;


namespace fast {

	struct DiffusionPrepareParams {		
		Dir dir = X_NEG;
		double d0 = 0.0;
		double d1 = 1.0;
		vec3 cellDim = vec3(1.0, 1.0, 1.0);
		bool verbose = false;
		PrimitiveType type = TYPE_FLOAT;
	};
	
	
	FAST_EXPORT std::unique_ptr<fast::LinearSystem> GenerateDiffusionLinearSystemHost(
		const DiffusionPrepareParams & params,
		ivec3 res, const uchar * mask,
		int numThreads = -1
	);

	FAST_EXPORT std::unique_ptr<fast::LinearSystem> GenerateDiffusionLinearSystemDevice(
		const DiffusionPrepareParams & params,
		CUDA_Volume * mask
		);


}