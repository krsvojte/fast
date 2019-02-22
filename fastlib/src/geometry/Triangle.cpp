#include "geometry/Triangle.h"

namespace fast {

	vec3 fast::Triangle::cross() const
	{
		const vec3 e[2] = {
			v[1] - v[0],
			v[2] - v[0]
		};
		return glm::cross(e[0], e[1]);
	}

	vec3 fast::Triangle::normal() const
	{
		return glm::normalize(cross());
	}

	float fast::Triangle::area() const
	{
		return glm::length(cross()) * 0.5f;
	}

	fast::Triangle fast::Triangle::flipped() const
	{
		return { v[0],v[2],v[1] };
	}

}
