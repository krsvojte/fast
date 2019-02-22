#pragma once

#include "VertexBuffer.h"

VertexBuffer<VertexData> getQuadVBO();

VertexBuffer<VertexData> getCubeVBO();

VertexBuffer<VertexData> getSphereVBO();


namespace fast {
	struct ConvexPolyhedron;
	struct TriangleMesh;
}

VertexBuffer<VertexData> getTriangleMeshVBO(const fast::TriangleMesh & cp, vec4 color);
