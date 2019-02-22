#include "PrimitivesVBO.h"

#include <fastlib/geometry/TriangleMesh.h>

#include <glm/gtc/type_ptr.hpp>

VertexBuffer<VertexData> getQuadVBO()
{
	VertexBuffer<VertexData> vbo;

	std::vector<VertexData> data = {
		VertexData({ -1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0,-1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0,1.0,0.1 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 })
	};
	vbo.setData(data.begin(), data.end());
	vbo.setPrimitiveType(GL_QUADS);

	return vbo;
}

VertexBuffer<VertexData> getCubeVBO()
{
	VertexBuffer<VertexData> ivbo;


	const std::vector<VertexData> data = {
		VertexData({ -1.0f, -1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, -1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, 1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, 1.0f, 1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, -1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, -1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ 1.0f, 1.0f, -1.0f },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 }),
		VertexData({ -1.0f, 1.0f, -1.0 },{ 0,0,0 },{ 0,0 },{ 0,0,0,0 })
	};


	const std::vector<unsigned int> indices = {
		0, 1, 2, 2, 3, 0,
		3, 2, 6, 6, 7, 3,
		7, 6, 5, 5, 4, 7,
		4, 0, 3, 3, 7, 4,
		0, 5, 1, 5, 0, 4,
		1, 5, 6, 6, 2, 1
	};


	std::vector<VertexData> tridata;

	for (auto i = 0; i < indices.size(); i += 3) {
		auto a = indices[i];
		auto b = indices[i + 1];
		auto c = indices[i + 2];
		tridata.push_back(data[a]);
		tridata.push_back(data[b]);
		tridata.push_back(data[c]);
	}


	ivbo.setData(tridata.begin(), tridata.end());
	//ivbo.setIndices<unsigned int>(indices.begin(), indices.end(), GL_UNSIGNED_INT);
	ivbo.setPrimitiveType(GL_TRIANGLES);

	return ivbo;
}

VertexBuffer<VertexData> getSphereVBO()
{


	VertexBuffer<VertexData> vbo;
	auto mesh = fast::generateSphere(1.0f, 32, 16);
	
	

	
	std::vector<VertexData> data;
	data.reserve(mesh.size() * 3);

	VertexData vd;
	vd.color[0] = 0.5f;
	vd.color[1] = 0.5f;
	vd.color[2] = 0.5f;
	vd.color[3] = 1.0f;
	vd.uv[0] = 0.0f;
	vd.uv[1] = 0.0f;

	for (auto t : mesh) {
		const auto N = t.normal();
		memcpy(&vd.normal, glm::value_ptr(N), 3 * sizeof(float));

		for (auto v : t.v) {
			memcpy(&vd.pos, glm::value_ptr(v), 3 * sizeof(float));
			data.push_back(vd);
		}
	}

	vbo.setPrimitiveType(GL_TRIANGLES);

	vbo.setData(data.begin(), data.end());

	return vbo;

}

VertexBuffer<VertexData> getTriangleMeshVBO(const fast::TriangleMesh & cp, vec4 color)
{
	VertexBuffer<VertexData> vbo;

	

	std::vector<vec3> normals(cp.vertices.size());

	std::vector<uint> indices;
	for (auto & f : cp.faces) {
		for (auto ind : f.vertices) {
			indices.push_back(ind);
			normals[ind] += f.normal;
		}
	}

	for (auto & n : normals) {
		n = glm::normalize(n);
	}


	{
		std::vector<VertexData> data;
		for(auto i=0; i < cp.vertices.size(); i++){
			const vec3 & v = cp.vertices[i];
			const vec3 & n = normals[i];
			VertexData vd;
			memcpy(vd.pos, &v, sizeof(vec3));
			memcpy(vd.normal, &n, sizeof(vec3));			
			memcpy(vd.color, &color, sizeof(vec4));			
			vd.uv[0] = 0.0f;
			vd.uv[1] = 0.0f;
			data.push_back(vd);
		}
		vbo.setData(data.begin(), data.end());
	}


	

	vbo.setIndices(indices.data(), indices.size(), GL_UNSIGNED_INT);
	vbo.setPrimitiveType(GL_TRIANGLES);



	return vbo;
}

