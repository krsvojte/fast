#pragma once

#include "TriangleMesh.h"

#include <vector>
#include <fstream>

namespace fast {
	
	struct GeometryObject;
	
	//Custom format
	FAST_EXPORT fast::TriangleMesh loadParticleMesh(const std::string & path);

	FAST_EXPORT std::vector<std::shared_ptr<fast::GeometryObject>> readPosFile(
		std::ifstream & stream, size_t index = 0, AABB trim = AABB::unit());

	FAST_EXPORT size_t getPosFileCount(std::ifstream & stream);

}