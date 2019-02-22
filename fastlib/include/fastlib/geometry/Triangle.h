#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/Types.h>

#include <array>

namespace fast {
		
	struct Triangle {
		std::array<vec3, 3> v;		

		/*
			Returns cross product of triangle's edges
		*/
		FAST_EXPORT vec3 cross() const;

		/*
			Returns unit normal vector of the triangle
		*/
		FAST_EXPORT vec3 normal() const;

		/*
			Returns area of the triangle
		*/
		
		FAST_EXPORT float area() const;

		/*
			Returns triangle with reverse winding order
		*/
		FAST_EXPORT Triangle flipped() const;

	};

}