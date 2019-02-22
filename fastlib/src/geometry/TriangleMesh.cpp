#include "geometry/TriangleMesh.h"

namespace fast {

	const float pi = static_cast<float>(std::acos(-1.0));

	vec3 sphToCart(float a, float b) {
		return {
			cos(a) * sin(b),
			sin(a) * sin(b),
			cos(b)
		};
	}


	FAST_EXPORT TriangleArray generateSphere(float radius, size_t polarSegments, size_t azimuthSegments)
	{

		TriangleArray mesh;

		const float stepX = (1.0f / polarSegments) * pi * 2.0f;
		const float stepY = (1.0f / azimuthSegments) * pi;

		for (int j = 0; j < azimuthSegments; j++) {
			const float b0 = j * stepY;
			const float b1 = (j + 1) * stepY;
			for (int i = 0; i < polarSegments; i++) {
				const float a0 = i * stepX;
				const float a1 = (i + 1) * stepX;

				const vec3 v[4] = {
					radius * sphToCart(a0, b0),
					radius * sphToCart(a0, b1),
					radius * sphToCart(a1, b0),
					radius * sphToCart(a1, b1)
				};

				mesh.push_back({ v[0],v[1],v[3] });
				mesh.push_back({ v[0],v[3],v[2] });
			}
		}

		return mesh;
	}

	FAST_EXPORT void TriangleMesh::recomputeNormals()
	{
		for (auto & f : faces) {
			Edge edgeA = { f.vertices[0], f.vertices[1] };
			Edge edgeB = { f.vertices[1], f.vertices[2] };
			vec3 edgeAvec = vertices[edgeA.x] - vertices[edgeA.y];
			vec3 edgeBvec = vertices[edgeB.x] - vertices[edgeB.y];
			f.normal = glm::normalize(glm::cross(edgeAvec, edgeBvec));
		}
	}


	FAST_EXPORT AABB TriangleMesh::bounds() const
	{
		AABB b;
		for (auto & v : vertices) {
			b.min = glm::min(b.min, v);
			b.max = glm::max(b.max, v);
		}

		return b;
	}

	FAST_EXPORT std::unique_ptr<Geometry> TriangleMesh::normalized(bool keepAspectRatio /*= true*/) const
	{
		auto b = bounds();

		Transform t;
		vec3 range = b.max - b.min;
		if (keepAspectRatio) {
			float maxRange = glm::max(range.x, glm::max(range.y, range.z));
			range = vec3(maxRange);
		}
		t.scale = { 1.0f / range.x, 1.0f / range.y, 1.0f / range.z };
		t.translation = -(b.min * t.scale) - ((b.max - b.min)* t.scale * 0.5f);

		return transformed(t);
	}

	FAST_EXPORT std::unique_ptr<Geometry> TriangleMesh::transformed(const Transform & t) const
	{
		auto newmesh = std::make_unique<TriangleMesh>(*this);

		for (auto & v : newmesh->vertices) {
			v = t.transformPoint(v);
		}

		newmesh->recomputeNormals();

		return newmesh;
	}

	FAST_EXPORT TriangleArray TriangleMesh::getTriangleArray() const
	{
		TriangleArray result(faces.size() * 3);

		for (auto i = 0; i < faces.size(); i++) {
			auto & f = faces[i];
			assert(f.vertices.size() == 3);
			result[i].v[0] = vertices[f.vertices[0]];
			result[i].v[1] = vertices[f.vertices[1]];
			result[i].v[2] = vertices[f.vertices[2]];
		}

		return result;
	}

}