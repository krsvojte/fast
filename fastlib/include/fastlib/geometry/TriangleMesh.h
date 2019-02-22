#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/geometry/Geometry.h>
#include <fastlib/geometry/Triangle.h>
#include <vector>

namespace fast {

	using TriangleArray = std::vector<Triangle>;



	FAST_EXPORT TriangleArray generateSphere(float radius = 1.0f, size_t polarSegments = 16, size_t azimuthSegments = 16);


	struct TriangleMesh : public Geometry {
		using Edge = ivec2;
		struct Face {
			std::vector<int> vertices;
			vec3 normal;
		};

		std::vector<vec3> vertices;
		std::vector<Face> faces;
		std::vector<Edge> edges;

		FAST_EXPORT virtual AABB bounds() const override;

		FAST_EXPORT virtual std::unique_ptr<Geometry> normalized(bool keepAspectRatio = true) const override;

		FAST_EXPORT virtual std::unique_ptr<Geometry> transformed(const Transform & t) const override;
		
		FAST_EXPORT void recomputeNormals();		

		FAST_EXPORT TriangleArray getTriangleArray() const;

	};

}