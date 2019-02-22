#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/Types.h>

#include <fastlib/utility/DataPtr.h>
#include <vector>


namespace fast {
		
	class Volume;
	

	FAST_EXPORT Volume getVolumeArea(
		const Volume & mask, 
		ivec3 res = ivec3(0,0,0),
		float isovalue = 0.5f
	);

	/*
		Returns VBO
	*/

	struct VolumeAreaMeshOutput {
		uint vbo;
		size_t Nverts;
		std::vector<CUDA_VBO::DefaultAttrib> data;
	};

	FAST_EXPORT VolumeAreaMeshOutput getVolumeAreaMesh(
		const Volume & mask,		
		bool openGLInterop = true,
		ivec3 res = ivec3(0, 0, 0),
		float isovalue = 0.5f
	);

	



}