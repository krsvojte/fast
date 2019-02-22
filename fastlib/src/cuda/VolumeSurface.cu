#include "utility/DataPtr.h"

#include "cuda/VolumeSurface.cuh"
#include "cuda/MCTable.cuh"

#include <assert.h>
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>




__device__ __constant__ uchar const_numVertsTable[256];
__device__ __constant__ uint const_edgeTable[256];
__device__ __constant__ uchar const_triTable[256 * 16];



struct VoxelCornerVals {
	float v[8];
};

struct VoxelCornerPos {
	float3 p[8];
};

#define TRIANGLE_THREADS 32

bool commitMCConstants() {
	if (cudaMemcpyToSymbol(const_numVertsTable,	&numVertsTable,
		sizeof(uchar) * 256,0,
		cudaMemcpyHostToDevice
	) != cudaSuccess) return false;

	if (cudaMemcpyToSymbol(const_edgeTable, &edgeTable,
		sizeof(uint) * 256, 0,
		cudaMemcpyHostToDevice
	) != cudaSuccess) return false;

	if (cudaMemcpyToSymbol(const_triTable, &triTable,
		sizeof(uchar) * 256 * 16, 0,
		cudaMemcpyHostToDevice
	) != cudaSuccess) return false;

	return true;
}



inline __device__ float getSmoothVolumeVal(cudaTextureObject_t tex, float x, float y, float z, float3 d) {

	return tex3D<float>(tex, x,y,z);

	/*const float3 pos = make_float3(x, y, z);	
	const float3 offsets[8] = {
		{ 0,0,0 },
		{ d.x,0,0 },
		{ d.x,d.y,0 },
		{ 0,d.y,0 },
		{ 0,0,d.z },
		{ d.x,0,d.z },
		{ d.x,d.y,d.z},
		{ 0,d.y,d.z }
	};

	float sum = 0.0;	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		float3 newPos = pos + offsets[i];
		sum += tex3D<float>(tex, newPos.x, newPos.y, newPos.z);
	}

	return sum / 8.0f;	*/
}

inline __device__ VoxelCornerVals getVoxelCornerVals(
	cudaTextureObject_t tex,
	float3 pos,
	float3 voxelSize,
	float smooth
) {
	
	VoxelCornerVals vals;
	float3 smoothOffset = voxelSize * smooth;

	//printf("Reading at pos %f %f %f\n", pos.x, pos.y, pos.z);
	vals.v[0] =getSmoothVolumeVal(tex, pos.x, pos.y, pos.z, smoothOffset);
	vals.v[1] =getSmoothVolumeVal(tex, pos.x + voxelSize.x, pos.y, pos.z, smoothOffset);
	vals.v[2] =getSmoothVolumeVal(tex, pos.x + voxelSize.x, pos.y + voxelSize.y, pos.z, smoothOffset);
	vals.v[3] =getSmoothVolumeVal(tex, pos.x, pos.y + voxelSize.y, pos.z, smoothOffset);

	vals.v[4] =getSmoothVolumeVal(tex, pos.x, pos.y, pos.z + voxelSize.z, smoothOffset);
	vals.v[5] =getSmoothVolumeVal(tex, pos.x + voxelSize.x, pos.y, pos.z + voxelSize.z, smoothOffset);
	vals.v[6] =getSmoothVolumeVal(tex, pos.x + voxelSize.x, pos.y + voxelSize.y, pos.z + voxelSize.z, smoothOffset);
	vals.v[7] =getSmoothVolumeVal(tex, pos.x, pos.y + voxelSize.y, pos.z + voxelSize.z, smoothOffset);
	
	return vals;
}

inline __device__ VoxelCornerPos getVoxelCornerPos(const float3& pos, const float3 &voxelSize){
	VoxelCornerPos v;
	v.p[0] = pos;
	v.p[1] = pos + make_float3(voxelSize.x, 0, 0);
	v.p[2] = pos + make_float3(voxelSize.x, voxelSize.y, 0);
	v.p[3] = pos + make_float3(0, voxelSize.y, 0);
	v.p[4] = pos + make_float3(0, 0, voxelSize.z);
	v.p[5] = pos + make_float3(voxelSize.x, 0, voxelSize.z);
	v.p[6] = pos + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v.p[7] = pos + make_float3(0, voxelSize.y, voxelSize.z);
	return v;
}

template <bool gt = false>
inline __device__ uint getCubeIndex(const VoxelCornerVals & vals, float isoValue) {

	uint cubeindex = 0;
	if (!gt) {
		cubeindex = uint(vals.v[0] < isoValue);
		cubeindex += uint(vals.v[1] < isoValue) * 2;
		cubeindex += uint(vals.v[2] < isoValue) * 4;
		cubeindex += uint(vals.v[3] < isoValue) * 8;
		cubeindex += uint(vals.v[4] < isoValue) * 16;
		cubeindex += uint(vals.v[5] < isoValue) * 32;
		cubeindex += uint(vals.v[6] < isoValue) * 64;
		cubeindex += uint(vals.v[7] < isoValue) * 128;
	}
	else {
		cubeindex = uint(vals.v[0] > isoValue);
		cubeindex += uint(vals.v[1] > isoValue) * 2;
		cubeindex += uint(vals.v[2] > isoValue) * 4;
		cubeindex += uint(vals.v[3] > isoValue) * 8;
		cubeindex += uint(vals.v[4] > isoValue) * 16;
		cubeindex += uint(vals.v[5] > isoValue) * 32;
		cubeindex += uint(vals.v[6] > isoValue) * 64;
		cubeindex += uint(vals.v[7] > isoValue) * 128;
	}
	return cubeindex;

}

__global__ void ___markVoxels(
	VolumeSurface_MCParams params,
	CUDA_Volume volume,
	uint * vertCount,
	uint * occupancy
) {

	VOLUME_IVOX_GUARD(params.res);
	const size_t i = _linearIndex(params.res, ivox);	
	

	const float3 voxelSize = make_float3(1.0f / params.res.x, 1.0f / params.res.y, 1.0f / params.res.z);
	const  float3 pos = make_float3(voxelSize.x * ivox.x, voxelSize.y * ivox.y, voxelSize.z * ivox.z);
	
	//Sample volume
	const VoxelCornerVals vals = getVoxelCornerVals(volume.tex, pos, voxelSize, params.smoothingOffset);

	//Calculate occupancy
	const uint cubeindex = getCubeIndex(vals, params.isovalue);	
	const uint numVerts = uint(const_numVertsTable[cubeindex]);

	/*if (ivox.x == 0 && ivox.y == 0 && ivox.z == 0) {
		printf("i %u, %f %f %f, |%f|%f|%f|%f||%f|%f|%f|%f| -> %u %u \n", uint(i), pos.x, pos.y, pos.z,
			vals.v[0], vals.v[1], vals.v[2], vals.v[3], vals.v[4], vals.v[5], vals.v[6], vals.v[7],
			cubeindex, numVerts
			);
	}*/

	vertCount[i] = numVerts;
	occupancy[i] = uchar(numVerts > 0);

}

__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	if (f1 == f0) {
		return p0;		
	}
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}


__device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{	
	return normalize(cross(*v1 - *v0, *v2 - *v0));
}


__device__ float triangleArea(float3 *v0, float3 *v1, float3 *v2) {
	return length(cross(*v1 - *v0, *v2 - *v0)) * 0.5f;
}


template <typename outType>
__global__ void ___generateTrianglesSurfaceArea(
	VolumeSurface_MCParams params,
	const CUDA_Volume in,	
	CUDA_Volume out	
){
	

	uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
	uint i = blockId * blockDim.x + threadIdx.x;

	const int3 ivox = posFromLinear(make_int3(params.res), i);
	
	if (ivox.x >= params.res.x || ivox.y >= params.res.y || ivox.z >= params.res.z) {
		return;
	}
	
	const float3 voxelSize = make_float3(1.0f / params.res.x, 1.0f / params.res.y, 1.0f / params.res.z);
	const float3 pos = make_float3(voxelSize.x * ivox.x, voxelSize.y * ivox.y, voxelSize.z * ivox.z);

	//Compute corner values
	const VoxelCornerVals vals = getVoxelCornerVals(in.tex, pos, voxelSize, params.smoothingOffset);
	const uint cubeindex = getCubeIndex(vals, params.isovalue);

	const VoxelCornerPos v = getVoxelCornerPos(pos, voxelSize);

	__shared__ float3 vertlist[12 * TRIANGLE_THREADS];
	vertlist[threadIdx.x] = vertexInterp(params.isovalue, v.p[0], v.p[1], vals.v[0], vals.v[1]);
	vertlist[TRIANGLE_THREADS + threadIdx.x] = vertexInterp(params.isovalue, v.p[1], v.p[2], vals.v[1], vals.v[2]);
	vertlist[(TRIANGLE_THREADS * 2) + threadIdx.x] = vertexInterp(params.isovalue, v.p[2], v.p[3], vals.v[2], vals.v[3]);
	vertlist[(TRIANGLE_THREADS * 3) + threadIdx.x] = vertexInterp(params.isovalue, v.p[3], v.p[0], vals.v[3], vals.v[0]);
	vertlist[(TRIANGLE_THREADS * 4) + threadIdx.x] = vertexInterp(params.isovalue, v.p[4], v.p[5], vals.v[4], vals.v[5]);
	vertlist[(TRIANGLE_THREADS * 5) + threadIdx.x] = vertexInterp(params.isovalue, v.p[5], v.p[6], vals.v[5], vals.v[6]);
	vertlist[(TRIANGLE_THREADS * 6) + threadIdx.x] = vertexInterp(params.isovalue, v.p[6], v.p[7], vals.v[6], vals.v[7]);
	vertlist[(TRIANGLE_THREADS * 7) + threadIdx.x] = vertexInterp(params.isovalue, v.p[7], v.p[4], vals.v[7], vals.v[4]);
	vertlist[(TRIANGLE_THREADS * 8) + threadIdx.x] = vertexInterp(params.isovalue, v.p[0], v.p[4], vals.v[0], vals.v[4]);
	vertlist[(TRIANGLE_THREADS * 9) + threadIdx.x] = vertexInterp(params.isovalue, v.p[1], v.p[5], vals.v[1], vals.v[5]);
	vertlist[(TRIANGLE_THREADS * 10) + threadIdx.x] = vertexInterp(params.isovalue, v.p[2], v.p[6], vals.v[2], vals.v[6]);
	vertlist[(TRIANGLE_THREADS * 11) + threadIdx.x] = vertexInterp(params.isovalue, v.p[3], v.p[7], vals.v[3], vals.v[7]);
	__syncthreads();

	uint numVerts = uint(const_numVertsTable[cubeindex]);

	outType totalArea = outType(0);
	for (int j = 0; j < numVerts; j += 3)
	{		

		float3 *vert[3];
		uint edge;
		edge = uint(const_triTable[(cubeindex * 16) + j]);
		vert[0] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];

		edge = uint(const_triTable[(cubeindex * 16) + j + 1]);
		vert[1] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];

		edge = uint(const_triTable[(cubeindex * 16) + j + 2]);
		vert[2] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];

		totalArea += outType(triangleArea(vert[0], vert[1], vert[2]));
	}

	write<outType>(out.surf, ivox, totalArea);

	

}

__global__ void ___generateTriangles(
	VolumeSurface_MCParams params,
	const CUDA_Volume volume,
	const uint * compacted,
	const uint * vertCountScan,	
	const size_t activeN,	
	fast::CUDA_VBO::DefaultAttrib * vbo	
) {

	uint blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	uint i = blockId * blockDim.x + threadIdx.x;
	
	if (i > activeN - 1) {
		return;		
	}

	uint voxelIndex = compacted[i];
	int3 vox = posFromLinear(make_int3(params.res), voxelIndex);
	int3 ivox = vox;
	

	//Recompute corner vals
	const float3 voxelSize = make_float3(1.0f / params.res.x, 1.0f / params.res.y, 1.0f / params.res.z);
	const float3 pos = make_float3(voxelSize.x * ivox.x, voxelSize.y * ivox.y, voxelSize.z * ivox.z);
	const VoxelCornerVals vals = getVoxelCornerVals(volume.tex, pos, voxelSize, params.smoothingOffset);
	const uint cubeindex = getCubeIndex(vals, params.isovalue);

	//Target position of the mesh
	const float3 targetVoxelSize = make_float3(params.dx,params.dy,params.dz);
	//Unit cube (0.5 to 0.5)
	const float3 targetPos = make_float3(targetVoxelSize.x * ivox.x, targetVoxelSize.y * ivox.y, targetVoxelSize.z * ivox.z) 
		- 0.5f * make_float3(targetVoxelSize.x * params.res.x, targetVoxelSize.y * params.res.y, targetVoxelSize.z * params.res.z);
	

	const VoxelCornerPos v = getVoxelCornerPos(targetPos, targetVoxelSize);	

	__shared__ float3 vertlist[12 * TRIANGLE_THREADS];	
	vertlist[threadIdx.x] = vertexInterp(params.isovalue, v.p[0], v.p[1], vals.v[0], vals.v[1]);
	vertlist[TRIANGLE_THREADS + threadIdx.x] = vertexInterp(params.isovalue, v.p[1], v.p[2], vals.v[1], vals.v[2]);
	vertlist[(TRIANGLE_THREADS * 2) + threadIdx.x] = vertexInterp(params.isovalue, v.p[2], v.p[3], vals.v[2], vals.v[3]);
	vertlist[(TRIANGLE_THREADS * 3) + threadIdx.x] = vertexInterp(params.isovalue, v.p[3], v.p[0], vals.v[3], vals.v[0]);
	vertlist[(TRIANGLE_THREADS * 4) + threadIdx.x] = vertexInterp(params.isovalue, v.p[4], v.p[5], vals.v[4], vals.v[5]);
	vertlist[(TRIANGLE_THREADS * 5) + threadIdx.x] = vertexInterp(params.isovalue, v.p[5], v.p[6], vals.v[5], vals.v[6]);
	vertlist[(TRIANGLE_THREADS * 6) + threadIdx.x] = vertexInterp(params.isovalue, v.p[6], v.p[7], vals.v[6], vals.v[7]);
	vertlist[(TRIANGLE_THREADS * 7) + threadIdx.x] = vertexInterp(params.isovalue, v.p[7], v.p[4], vals.v[7], vals.v[4]);
	vertlist[(TRIANGLE_THREADS * 8) + threadIdx.x] = vertexInterp(params.isovalue, v.p[0], v.p[4], vals.v[0], vals.v[4]);
	vertlist[(TRIANGLE_THREADS * 9) + threadIdx.x] = vertexInterp(params.isovalue, v.p[1], v.p[5], vals.v[1], vals.v[5]);
	vertlist[(TRIANGLE_THREADS * 10) + threadIdx.x] = vertexInterp(params.isovalue, v.p[2], v.p[6], vals.v[2], vals.v[6]);
	vertlist[(TRIANGLE_THREADS * 11) + threadIdx.x] = vertexInterp(params.isovalue, v.p[3], v.p[7], vals.v[3], vals.v[7]);
	__syncthreads();
	
	uint numVerts = uint(const_numVertsTable[cubeindex]);
	for (int j = 0; j < numVerts; j += 3)
	{
		uint index = vertCountScan[voxelIndex] + j;

		float3 *vert[3];
		uint edge;
		edge = uint(const_triTable[(cubeindex * 16) + j]);				
		vert[0] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];
		
		edge = uint(const_triTable[(cubeindex * 16) + j + 1]);
		vert[1] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];
		
		edge = uint(const_triTable[(cubeindex * 16) + j + 2]);
		vert[2] = &vertlist[(edge*TRIANGLE_THREADS) + threadIdx.x];

		float3 n = calcNormal(vert[0], vert[1], vert[2]);
		
		#pragma unroll
		for (int k = 0; k < 3; k++) {
			fast::CUDA_VBO::DefaultAttrib & att = vbo[index + k];
			
			att.pos[0] = (*vert[k]).x;
			att.pos[1] = (*vert[k]).y;
			att.pos[2] = (*vert[k]).z;			

			att.normal[0] = n.x;
			att.normal[1] = n.y;
			att.normal[2] = n.z;			
			const float3 c = make_float3(1.0f);		

			att.color[0] = c.x;
			att.color[1] = c.y;
			att.color[2] = c.z;
			att.color[3] = 1.0;
		
		}		
	}



}


__global__ void ___compactVoxels(
	const uint3 res,
	const uint * occupancy,
	const uint * occupancyScan,		
	uint * compacted
) {
	VOLUME_IVOX_GUARD(res);
	size_t i = _linearIndex(res, ivox);

	if (occupancy[i]) {
		compacted[occupancyScan[i]] = i;
	}
}

template <typename T>
T scanAndCount(T * input, T * scanned, size_t N) {
	thrust::exclusive_scan(thrust::device_ptr<T>(input),
		thrust::device_ptr<T>(input + N),
		thrust::device_ptr<T>(scanned));

	uint lastInput, lastScan;
	cudaMemcpy(&lastInput, (input + N - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(&lastScan, (scanned + N - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	return lastInput + lastScan;
}



void VolumeSurface_MarchingCubesMesh(const CUDA_Volume & input,
	const VolumeSurface_MCParams & params,
	bool openGLInterop,
	uint * vboOut,
	size_t * NvertsOut,
	std::vector<fast::CUDA_VBO::DefaultAttrib> & dataOut
)
{
	

	
	assert(input.type == TYPE_UCHAR);

	commitMCConstants();	

	size_t N = params.res.x * params.res.y * params.res.z;

	if (*vboOut && *NvertsOut) {
		*NvertsOut = 0;
	}


	uint * vertCount;
	uint * vertCountScan;
	uint * occupancy;
	uint * occupancyScan;
	uint * compacted;

	cudaMalloc(&vertCount, N * sizeof(uint));
	cudaMalloc(&vertCountScan, N * sizeof(uint));
	cudaMalloc(&occupancy, N * sizeof(uint));
	cudaMalloc(&occupancyScan, N * sizeof(uint));
	cudaMalloc(&compacted, N * sizeof(uint));

	auto freeResouces = [=]{
		cudaFree(vertCount);
		cudaFree(vertCountScan);
		cudaFree(occupancy);
		cudaFree(occupancyScan);
		cudaFree(compacted);
	};


	//Find occupied voxels and number of triangles
	{
		BLOCKS3D(8, params.res);
		___markVoxels<<< numBlocks, block>>>(
			params, input, vertCount, occupancy
		);
	}

	

	//Perform scan on occupancy -> counts number of occupied voxels
	uint activeN = scanAndCount(occupancy, occupancyScan, N);
	
	
	
	if (activeN == 0) {		
		freeResouces();
		return;
	}

	//Compact
	{
		BLOCKS3D(8, params.res);
		___compactVoxels << < numBlocks, block >> > (params.res, occupancy, occupancyScan, compacted);
	}

	//Scan vert count
	uint totalVerts = scanAndCount(vertCount, vertCountScan, N);	


	//Allocate output buffer (either using interop or directly by cudaMalloc)
	std::unique_ptr<fast::CUDA_VBO> cudaVBO;
	fast::CUDA_VBO::DefaultAttrib * outputPtr = nullptr;
	if (openGLInterop) {
		cudaVBO = std::make_unique<fast::CUDA_VBO>(
			fast::createMappedVBO(totalVerts * sizeof(fast::CUDA_VBO::DefaultAttrib))
		);	
		outputPtr = static_cast<fast::CUDA_VBO::DefaultAttrib *>(cudaVBO->getPtr());
	}
	else {
		cudaMalloc(&outputPtr, totalVerts * sizeof(fast::CUDA_VBO::DefaultAttrib));
	}
	

	{
		dim3 grid = dim3((activeN + TRIANGLE_THREADS - 1) / TRIANGLE_THREADS, 1, 1);
		while (grid.x > 65535) {
			grid.x /= 2;
			grid.y *= 2;
		}

		___generateTriangles<< <grid, TRIANGLE_THREADS >>>(
			params, input, compacted, vertCountScan, activeN, outputPtr
			);

	}


	/*
		Return either vbo or data directly
	*/
	if (openGLInterop) {
		*vboOut = cudaVBO->getVBO();		
		outputPtr = nullptr;
	}
	else {
		dataOut.resize(totalVerts);
		cudaMemcpy(dataOut.data(), outputPtr, totalVerts * sizeof(fast::CUDA_VBO::DefaultAttrib), cudaMemcpyDeviceToHost);			
		cudaFree(outputPtr);
	}
	*NvertsOut = totalVerts;
	
	freeResouces();	

}

void VolumeSurface_MarchingCubesArea(
	const CUDA_Volume & input, 
	const VolumeSurface_MCParams & params,
	CUDA_Volume & output
)
{	
	commitMCConstants();

	assert(output.type == TYPE_FLOAT || output.type == TYPE_DOUBLE);

	{
		size_t N = params.res.x * params.res.y * params.res.z;
		dim3 grid = dim3((N + TRIANGLE_THREADS - 1) / TRIANGLE_THREADS, 1, 1);
		while (grid.x > 65535) {
			grid.x /= 2;
			grid.y *= 2;
		}

		if(output.type == TYPE_FLOAT)
			___generateTrianglesSurfaceArea<float><< <grid, TRIANGLE_THREADS >> > (
				params, input, output
				);
		else if(output.type == TYPE_DOUBLE)
			___generateTrianglesSurfaceArea<double> << <grid, TRIANGLE_THREADS >> > (
				params, input, output
				);
	}

	return;
}
