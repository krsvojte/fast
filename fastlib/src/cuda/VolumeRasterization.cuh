#pragma once

#include <cuda_runtime.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>

#include "Volume.cuh"

#include <vector>


void Volume_Rasterize(
	const float * meshTriangles, size_t triangleN,
	const float * transformMatrices4x4, size_t instanceN,
	CUDA_Volume & output
);

struct Volume_CollisionPair {
	size_t i,j;
};

std::vector<Volume_CollisionPair> Volume_AABB_Collisions(
	float * aabbs,
	size_t N,
	uint3 res
);