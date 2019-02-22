#include "VolumeRasterization.cuh"


#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <limits.h>
#include <float.h>

struct Triangle {
	float3 v[3];
}; 

struct BB {
	float3 minBB;
	float3 maxBB;
};

__device__ bool rayTriangleIntersects(const float3 v1, const float3 v2, const float3 v3, const float3 r_o, const float3 r_d, float &dist_to_surface)
{
	// Möller–Trumbore intersection algorithm
	// http://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm

	float3 e1, e2;  //Edge1, Edge2
	float3 P, Q, T;
	float det, inv_det, u, v;
	float t;
	float EPSILON = 0.00001f;

	//Find vectors for two edges sharing V1
	e1 = v2 - v1;
	e2 = v3 - v1;

	//Begin calculating determinant - also used to calculate u parameter
	P = cross(r_d, e2);

	//if determinant is near zero, ray lies in plane of triangle
	det = dot(e1, P);

	//NOT CULLING
	if (det > -EPSILON && det < EPSILON) return false;
	inv_det = 1.f / det;

	//calculate distance from V1 to ray origin
	T = r_o - v1;

	//Calculate u parameter and test bound
	u = dot(T, P) * inv_det;

	//The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f) return false;

	//Prepare to test v parameter
	Q = cross(T, e1);

	//Calculate V parameter and test bound
	v = dot(r_d, Q) * inv_det;

	//The intersection lies outside of the triangle
	if (v < 0.f || u + v  > 1.f) return false;

	t = dot(e2, Q) * inv_det;

	if (t > EPSILON)
	{
		//ray intersection
		dist_to_surface = t;
		return true;
	}

	// No hit, no win
	return false;
}



__device__ void bbox(const Triangle & tri, float3 * minBB, float3 * maxBB) {	
	*minBB = fminf(fminf(tri.v[0], tri.v[1]), tri.v[2]);
	*maxBB = fmaxf(fmaxf(tri.v[0], tri.v[1]), tri.v[2]);
}


__global__ void ___rasterizeKernel(
	uint3 res,
	uint3 resPlane,
	uint3 begin,
	uint3 end,
	int dir,
	Triangle * triangles, uint triangleN,
	CUDA_Volume output
){
	VOLUME_VOX_GUARD(resPlane);

	uint3 dirVox = begin;
	_at<uint>(dirVox, (dir + 1) % 3) += vox.x;
	_at<uint>(dirVox, (dir + 2) % 3) += vox.y;		
	

	float3 rayOrigin = make_float3(dirVox.x / float(res.x), dirVox.y / float(res.y), dirVox.z / float(res.z));
	float3 rayDir = make_float3(0);
	_at<float>(rayDir, dir) = 1.0f;
	_at<float>(rayOrigin, dir) = -1.0f; //make sure is far outside of the domain

	//Assumes convex mesh
	bool isects[2] = { false, false };
	float isectsD[2] = { 0, 0 };	
	int cntIsect = 0;
	for (auto i = 0; i < triangleN; i++) {
		float d = 0;
		bool isect = rayTriangleIntersects(triangles[i].v[0], triangles[i].v[1], triangles[i].v[2], rayOrigin, rayDir, d);
		if (isect) {
			if (!isects[0]) {
				isects[0] = true;
				isectsD[0] = d;
			}
			else if (!isects[1]) {
				isects[1] = true;
				isectsD[1] = d;
			}
			else {
				
			}
			cntIsect++;
		}
	}

	//No intersection
	if (!isects[0] && !isects[1])
		return;


	uint3 beginRaster = dirVox;
	uint3 endRaster = dirVox;
	_at<uint>(endRaster, dir) = _at<uint>(end, dir);	

	if (isects[0] && isects[1]) {
		float minIsect = fminf(isectsD[0], isectsD[1]) + _at<float>(rayOrigin, dir);
		float maxIsect = fmaxf(isectsD[0], isectsD[1]) + _at<float>(rayOrigin, dir);
		_at<uint>(beginRaster, dir) = uint(minIsect * _at<uint>(res, dir));
		_at<uint>(endRaster, dir) =  uint(ceilf(maxIsect * _at<uint>(res, dir)));


		_at<uint>(beginRaster, dir) = min(_at<uint>(beginRaster, dir), _at<uint>(end, dir));
		_at<uint>(endRaster, dir) = min(_at<uint>(endRaster, dir), _at<uint>(end, dir));

	}
	else if (!isects[0]) {
		float minIsect = isects[1];
		_at<uint>(beginRaster, dir) = uint((_at<float>(rayOrigin, dir) + minIsect) * _at<uint>(res, dir));
	}
	else /*if (!isects[1])*/ {
		float minIsect = isects[0];
		_at<uint>(beginRaster, dir) = uint((_at<float>(rayOrigin, dir) + minIsect) * _at<uint>(res, dir));
	}

	uint3 rasterVox = beginRaster;
	while (_at<uint>(rasterVox, dir) < _at<uint>(endRaster, dir)) {
		write<uchar>(output.surf, rasterVox, 255);
		_at<uint>(rasterVox, dir)++;
	}	

};

void Volume_Rasterize(
	const float * templateTriangles, size_t templateTriangleN,
	const float * transformMatrices4x4, size_t instanceN,
	CUDA_Volume & output
)
{
	const uint3 res = output.res;

	std::vector<BB> bbs(instanceN);

	thrust::device_vector<Triangle> triangles_d(templateTriangleN);
	thrust::host_vector<Triangle> triangles_h;

	int totalRows = 0;
	for (int ii = 0; ii < instanceN; ii++) {
		const matrix4x4 & M = ((matrix4x4*)transformMatrices4x4)[ii];
		bbs[ii].minBB = make_float3(FLT_MAX);
		bbs[ii].maxBB = make_float3(-FLT_MAX);

		triangles_h.clear();

		//Transform and calculate bounding box
		for (int ti = 0; ti < templateTriangleN; ti++) {
			const Triangle & t = ((Triangle*)templateTriangles)[ti];							
			Triangle t_t;
			for (int k = 0; k < 3; k++) {			
				t_t.v[k] = M.mulPt(t.v[k]);
				bbs[ii].minBB = fminf(bbs[ii].minBB, t_t.v[k]);
				bbs[ii].maxBB = fmaxf(bbs[ii].maxBB, t_t.v[k]);
			}
			triangles_h.push_back(t_t);
		}	

		triangles_d = triangles_h;


		BB & bb = bbs[ii];
		int3 IminBB = make_int3(int(bb.minBB.x * res.x), int(bb.minBB.y * res.y), int(bb.minBB.z * res.z));
		int3 ImaxBB = make_int3(int(ceilf(bb.maxBB.x * res.x)), int(ceilf(bb.maxBB.y * res.y)), int(ceilf(bb.maxBB.z * res.z)));
		IminBB -= make_int3(1);
		ImaxBB += make_int3(1);

		IminBB = clamp(IminBB, make_int3(0), make_int3(res));
		ImaxBB = clamp(ImaxBB, make_int3(0), make_int3(res));

		int3 bbdim = ImaxBB - IminBB;
		if (bbdim.x == 0 || bbdim.y == 0 || bbdim.z == 0)
			continue;

		int maxDim = min(bbdim.x, min(bbdim.y, bbdim.z));
		int minDimIndex = 0;
		if (bbdim.y == maxDim) minDimIndex = 1;
		else if (bbdim.z == maxDim) minDimIndex = 2;

		
		uint3 resPlane = make_uint3(
			((int*)&bbdim)[(minDimIndex + 1) % 3],
			((int*)&bbdim)[(minDimIndex + 2) % 3],
			1
		);
			

		BLOCKS3D_INT3(8, 8, 1, resPlane);

		totalRows += resPlane.x * resPlane.y;

		___rasterizeKernel << <numBlocks, block >> > (
			res,
			resPlane,
			make_uint3(IminBB),
			make_uint3(ImaxBB),
			minDimIndex,
			thrust::raw_pointer_cast(&triangles_d.front()),
			uint(triangles_d.size()),
			output
		);

		
	}	



	


}

/************************************************************************/


void __global__ ___AABBRasterizeKernel(
	const BB * bbs, 
	const size_t N, 
	const uint3 res,
	uint * counts,
	uint * indices
){
	
	uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
	uint i = blockId * blockDim.x + threadIdx.x;
	if (i > N) return;

	const BB bb = bbs[i];

	int3 IminBB = make_int3(int(bb.minBB.x * res.x), int(bb.minBB.y * res.y), int(bb.minBB.z * res.z));
	int3 ImaxBB = make_int3(int(ceilf(bb.maxBB.x * res.x)), int(ceilf(bb.maxBB.y * res.y)), int(ceilf(bb.maxBB.z * res.z)));
	IminBB = clamp(IminBB, make_int3(0), make_int3(res));
	ImaxBB = clamp(ImaxBB, make_int3(0), make_int3(res));

	/*if (i < 2) {
		printf("%u: %d %d %d, %d %d %d\n", i, IminBB.x, IminBB.y, IminBB.z, ImaxBB.x, ImaxBB.y, ImaxBB.z);
	}*/
	
	

	for (int z = IminBB.x; z < ImaxBB.x; z++) {
		for (int y = IminBB.x; y < ImaxBB.x; y++) {
			for (int x = IminBB.x; x < ImaxBB.x; x++) {
				int3 pos = make_int3(x, y, z);
				size_t k = _linearIndex(res, pos);
				//printf("%d %d %d", pos.x, pos.y, pos.z);
				//uint index = atomicAdd(counts + k, 1);
				atomicAdd(counts + k, 1);

			}
		}
	}


}

std::vector<Volume_CollisionPair> Volume_AABB_Collisions(float * aabbs, size_t N, uint3 res)
{
	
	size_t gridN = res.x * res.y * res.z;
	thrust::device_vector<uint> counts_d(gridN,0);
	
	
	thrust::device_vector<BB> bb_d(N);
	cudaMemcpy(thrust::raw_pointer_cast(&bb_d.front()), aabbs, N * sizeof(BB), cudaMemcpyHostToDevice);
	thrust::device_vector<uint> indices_d(N);


	std::vector<Volume_CollisionPair> result;

	int perBlock = 4;
	dim3 grid = dim3((N + perBlock - 1) / perBlock, 1, 1);
	while (grid.x > 65535) {
		grid.x /= 2;
		grid.y *= 2;
	}

	___AABBRasterizeKernel << <grid, perBlock>> > (		
		thrust::raw_pointer_cast(&bb_d.front()),
		N, 
		res,
		thrust::raw_pointer_cast(&counts_d.front()),
		thrust::raw_pointer_cast(&indices_d.front())
		);

	thrust::host_vector<uint> counts_h = counts_d;
	uint * counts_ptr = counts_h.data();


	uint totalCount = thrust::reduce(counts_d.begin(), counts_d.end(), 0);
	
	printf("res: %u, count: %u\n", res.x, totalCount);

	return result;
}



/*__global__ void ___triangleToVoxelCount(
	uint3 res,
	const Triangle * triTemplate, 	
	uint triTemplateN,
	const matrix4x4 * transforms,
	uint transformsN,
	Triangle * triangles,
	uint * counts)
{
	uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
	uint i = blockId * blockDim.x + threadIdx.x;

	if (i > triTemplateN*transformsN)
		return;

	uint transformIndex = i / triTemplateN;
	uint inTemplateIndex = i % triTemplateN;

	//Load triangle, transform it and save it
	Triangle triangle;
	{
		matrix4x4 transform = transforms[transformIndex];
		triangle = triTemplate[inTemplateIndex];

		triangle.v[0] = transform.mulPt(triangle.v[0]);
		triangle.v[1] = transform.mulPt(triangle.v[1]);
		triangle.v[2] = transform.mulPt(triangle.v[2]);
		triangles[i] = triangle;
	}

	//Get bounding box
	float3 minBB, maxBB;
	bbox(triangle, &minBB, &maxBB);

	//Calculate rasterization bounds
	int3 IminBB = make_int3(int(minBB.x * res.x), int(minBB.y * res.y), int(minBB.z * res.z));
	int3 ImaxBB = make_int3(int(ceilf(maxBB.x * res.x)), int(ceilf(maxBB.y * res.y)), int(ceilf(maxBB.z * res.z)));
	IminBB = clamp(IminBB, make_int3(0), make_int3(res));
	ImaxBB = clamp(ImaxBB, make_int3(0), make_int3(res));

	//Calculate triangle vs voxel intersections
	for (int z = IminBB.x; z < ImaxBB.x; z++) {
		for (int y = IminBB.x; y < ImaxBB.x; y++) {
			for (int x = IminBB.x; x < ImaxBB.x; x++) {

				
			}
		}
	}



	

	





	//for bb
	//counts[xyz] atomicadd

	
}


#define CPU_DEBUG


void Volume_Rasterize(
	float * templateTriangles, size_t templateTriangleN,
	float * transformMatrices4x4, size_t instanceN,
	CUDA_Volume & output
)
{
	const int maxConstantTri = 65535 / sizeof(Triangle);
	printf("Max constant triangles: %d\n", maxConstantTri);

	thrust::device_vector<Triangle> triTemplate(templateTriangleN);
#ifdef CPU_DEBUG 
	thrust::host_vector<Triangle> triTemplate_cpu(templateTriangleN);
#endif
	cudaMemcpy(
		thrust::raw_pointer_cast(&triTemplate.front()),
		templateTriangles,
		sizeof(Triangle) * templateTriangleN,
		cudaMemcpyHostToDevice
	);

	thrust::device_vector<matrix4x4> transforms(instanceN);
#ifdef CPU_DEBUG 
	thrust::host_vector<matrix4x4> transforms_cpu(instanceN);
#endif
	cudaMemcpy(
		thrust::raw_pointer_cast(&transforms.front()),
		transformMatrices4x4,
		sizeof(matrix4x4) * instanceN,
		cudaMemcpyHostToDevice
	);


	size_t triangleN = templateTriangleN * instanceN;

	thrust::device_vector<Triangle> triangles(triangleN);
	thrust::device_vector<uint> triangleInVoxelIndex(triangleN);
#ifdef CPU_DEBUG 
	thrust::host_vector<Triangle> triangles_cpu(instanceN);
	thrust::host_vector<uint> triangleInVoxelIndex_cpu(triangleN);
#endif

	
	uint3 res = output.res;
	size_t N = res.x*res.y*res.y;
	thrust::device_vector<uint> counts(N);
	thrust::device_vector<uint> countScan(N);
#ifdef CPU_DEBUG 
	thrust::host_vector<uint> counts_cpu(N);
	thrust::host_vector<uint> countScan_cpu(N);
#endif

	{
		int perBlock = 512;
		dim3 grid = dim3(roundDiv(triangleN, perBlock), 1, 1);
		while (grid.x > 65535) {
			grid.x /= 2;
			grid.y *= 2;
		}		

		___triangleToVoxelCount << <grid, perBlock >> > (
			res,
			thrust::raw_pointer_cast(&triTemplate.front()),			
			triTemplate.size(),
			thrust::raw_pointer_cast(&transforms.front()),
			transforms.size(),
			thrust::raw_pointer_cast(&triangles.front()),
			thrust::raw_pointer_cast(&counts.front())
		);

	}

#ifdef CPU_DEBUG 
	triTemplate_cpu = triTemplate;
	triangles_cpu = triangles;
	transforms_cpu = transforms;
	triangleInVoxelIndex_cpu = triangleInVoxelIndex;
	counts_cpu = counts;
	countScan_cpu = countScan;

	Triangle *ttemplate = triTemplate_cpu.data();
	Triangle *t = triangles_cpu.data();
	matrix4x4 * transf = transforms_cpu.data();
	uint * tInVoxel = triangleInVoxelIndex_cpu.data();
	uint * cnts = counts_cpu.data();
	uint * cntsScan = countScan_cpu.data();

#endif

	


#ifdef CPU_DEBUG 
	char b;
	b = 0;

#endif


}*/
