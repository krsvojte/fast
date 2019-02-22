#pragma once

#include <distfun/distfun.hpp>
#include <fastlib/utility/RandomGenerator.h>
#include <fastlib/FastLibDef.h>

namespace fast {

	class Volume;

	using SDFArray = std::vector<distfun::Primitive>;


	SDFArray filterSDFByAABB(
		const SDFArray & arr, 
		const distfun::AABB & domain
	);
	

	distfun::DistProgram SDFToProgram(
		const SDFArray & state,
		const distfun::AABB * inversionDomain = nullptr,
		const distfun::Primitive * intersectionPrimitive = nullptr
	);


	void SDFRasterize(
		const SDFArray & arr,
		const distfun::AABB & domain,
		Volume & volume,
		bool invert = false,
		bool commitToGPU = true
	);

	float SDFVolume(
		const SDFArray & arr,
		const distfun::AABB & domain,
		int maxDepth = 4,
		bool onDevice = false
	);


	
	std::vector<distfun::vec4> SDFElasticity(
		const SDFArray & arr,
		const distfun::AABB & domain,
		RNGUniformFloat & rng,
		int maxDepth = 3,
		bool onDevice = false
	);
	

	std::vector<float> SDFPerParticleOverlap(
		const SDFArray & arr,
		const distfun::AABB & domain,
		int maxDepth = 3
	);

}