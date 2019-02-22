#pragma once

#include <fastlib/FastLibDef.h>

#include <vector>
#include <memory>

namespace fast {


	
	class Volume;
	struct GeometryObject;


	FAST_EXPORT void rasterize(
		const std::vector<std::shared_ptr<GeometryObject>> & objects,
		Volume & output
	);

	FAST_EXPORT void rasterize(
		const float * meshTriangles, size_t triangleN,
		const float * transformMatrices4x4, size_t instanceN,
		Volume & output
	);

}