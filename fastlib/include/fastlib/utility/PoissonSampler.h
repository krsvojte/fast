#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/geometry/AABB.h>


#include <vector>

namespace fast {


	std::vector<vec3> poissonSampler(float r, size_t N, AABB domain, int k = 30);

}