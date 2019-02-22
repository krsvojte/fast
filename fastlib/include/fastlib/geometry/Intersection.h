#pragma once


#include <fastlib/FastLibDef.h>

/*
	bool isectTestXY(); .. only tests
	IsectXYResult isectXY(); .. returns struct with details
*/

namespace fast {

	struct TriangleMesh;
	struct Geometry;

	FAST_EXPORT bool isectTestConvexMesh(const TriangleMesh & a, const TriangleMesh & b);


	FAST_EXPORT bool isectTest(const Geometry & a, const Geometry & b);
	



}
