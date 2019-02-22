#include "geometry/Intersection.h"

#include "geometry/TriangleMesh.h"
#include "geometry/Geometry.h"


namespace fast {


	FAST_EXPORT bool isectTestConvexMesh(const TriangleMesh & a, const TriangleMesh & b)
	{

		const auto whichSide = [](const std::vector<vec3> & vertices, const vec3 & D, const vec3 & P){
			int positive = 0;
			int negative = 0;

			for (auto & vertex : vertices) {
				float t = glm::dot(D, vertex - P);
				if (t > 0.0f)
					positive++;
				else if (t < 0.0f)
					negative++;

				if (positive && negative)
					return 0;
			}

			return positive ? 1 : -1;
		};

		for (auto & face : a.faces) {
			auto D = face.normal;

			if (whichSide(b.vertices, D, a.vertices[face.vertices.front()]) > 0)
				return false;

		}

		for (auto & otherFace : b.faces) {
			auto Dother = otherFace.normal;

			if (whichSide(a.vertices, Dother, b.vertices[otherFace.vertices.front()]) > 0)
				return false;
		}

		for (auto & edge : a.edges) {
			auto edgeVec = a.vertices[edge[0]] - a.vertices[edge[1]];
			for (auto & otherEdge : b.edges) {
				auto otherEdgeVec = b.vertices[otherEdge[0]] - b.vertices[otherEdge[1]];
								
				auto D = glm::normalize(glm::cross(edgeVec, otherEdgeVec));

				int side0 = whichSide(a.vertices, D, a.vertices[edge[0]]);
				if (side0 == 0)
					continue;

				int side1 = whichSide(b.vertices, D, a.vertices[edge[0]]);
				if (side1 == 0)
					continue;

				if (side0*side1 < 0)
					return false;

			}
		}

		return true;
	}

	

	FAST_EXPORT bool isectTest(const Geometry & a, const Geometry & b)
	{
		const TriangleMesh * tm0 = dynamic_cast<const TriangleMesh*>(&a);

		if (tm0) {		
			const TriangleMesh * tm1 = dynamic_cast<const TriangleMesh*>(&b);
			if (tm1) {
				return isectTestConvexMesh(*tm0, *tm1);
			}
			else {
				assert("Unsupported collision test");
			}		
		}
		else {
			assert("Unsupported collision test");
		}
		return false;
	}

}
