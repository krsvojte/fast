#pragma once

#include <distfun/distfun.hpp>
#include <fastlib/utility/RandomGenerator.h>
#include <fastlib/FastLibDef.h>

namespace fast {

	class Volume;

	using SDFArray = std::vector<distfun::sdPrimitive>;

	FAST_EXPORT bool SDFSave(const SDFArray & arr, const std::string & filename);
	FAST_EXPORT SDFArray SDFLoad(const std::string & filename, std::function<const void *(const distfun::sdGridParam &tempGrid)> gridCallback);


	SDFArray filterSDFByAABB(
		const SDFArray & arr, 
		const distfun::sdAABB & domain
	);
	

	distfun::sdProgram SDFToProgram(
		const SDFArray & state,
		const distfun::sdAABB * inversionDomain = nullptr,
		const distfun::sdPrimitive * intersectionPrimitive = nullptr
	);


	void SDFRasterize(
		const SDFArray & arr,
		const distfun::sdAABB & domain,
		Volume & volume,
		bool invert = false,
		bool commitToGPU = true,
		bool overlap = false
	);

	float SDFVolume(
		const SDFArray & arr,
		const distfun::sdAABB & domain,
		int maxDepth = 4,
		bool onDevice = false
	);


	
	std::vector<distfun::vec4> SDFElasticity(
		const SDFArray & arr,
		const distfun::sdAABB & domain,
		RNGUniformFloat & rng,
		int maxDepth = 3,
		bool onDevice = false
	);
	

	std::vector<float> SDFPerParticleOverlap(
		const SDFArray & arr,
		const distfun::sdAABB & domain,
		int maxDepth = 3
	);

}