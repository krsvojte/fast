#include "Volume.cuh"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>


	



__global__ void kernelErode(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut) {
	
	VOLUME_VOX_GUARD(res);

	float vals[6];	
	surf3Dread(&vals[0], surfIn, ((vox.x - 1 + res.x) % res.x) * sizeof(float), vox.y, vox.z);
	surf3Dread(&vals[1], surfIn, ((vox.x + 1) % res.x) * sizeof(float), vox.y, vox.z);
	surf3Dread(&vals[2], surfIn, vox.x * sizeof(float), (vox.y - 1 + res.y) % res.y , vox.z);
	surf3Dread(&vals[3], surfIn, vox.x * sizeof(float), (vox.y + 1) % res.y, vox.z);
	surf3Dread(&vals[4], surfIn, vox.x * sizeof(float), vox.y, (vox.z - 1 + res.z) % res.z);
	surf3Dread(&vals[5], surfIn, vox.x * sizeof(float), vox.y, (vox.z + 1) % res.z);

	float valCenter;
	surf3Dread(&valCenter, surfIn, vox.x * sizeof(float), vox.y, vox.z);

	float newVal = valCenter;
	bool isInside = ((vals[0] > 0.0f) && (vals[1] > 0.0f) && (vals[2] > 0.0f) && 
					(vals[3] > 0.0f) && (vals[4] > 0.0f) && (vals[5] > 0.0f));
	
	if (!isInside) {
		newVal = 0.0f;
	}	

	surf3Dwrite(newVal, surfOut, vox.x * sizeof(float), vox.y, vox.z);
}

void launchErodeKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut){

	uint3 block = make_uint3(8, 8, 8);
	uint3 numBlocks = make_uint3(
		(res.x / block.x) + 1,
		(res.y / block.y) + 1,
		(res.z / block.z) + 1
	);

	kernelErode << <numBlocks, block >> > (res, surfIn, surfOut);
}



template <int blockSize, int apron>
__global__ void kernelHeat(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut) {

	static_assert(apron * 2 < blockSize, "Apron must be less than blockSize / 2");
	const int N = blockSize;
	const int3 tid = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);

	//Sliding window of blockdim - 2*apron size
	const int3 vox = make_int3(
		blockIdx.x * (blockDim.x - 2*apron), 
		blockIdx.y * (blockDim.y - 2*apron), 
		blockIdx.z * (blockDim.z - 2*apron)
	) + tid - make_int3(apron);

	//Toroidal boundaries
	const int3 voxToroid = make_int3((vox.x + res.x) % res.x, (vox.y + res.y) % res.y, (vox.z + res.z) % res.z);
	
	//Read whole block into shared memory
	__shared__ float ndx[N][N][N];
	surf3Dread(
		&ndx[tid.x][tid.y][tid.z],
		surfIn,
		voxToroid.x * sizeof(float), voxToroid.y, voxToroid.z
	);
	__syncthreads();
	
	//Skip apron voxels
	if (tid.x < apron || tid.x >= N - apron ||
		tid.y < apron || tid.y >= N - apron ||
		tid.z < apron || tid.z >= N - apron ) return;

	//Skip outside voxels
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ||
		vox.x < 0 || vox.y < 0 || vox.z < 0)	return;

	/////////// Compute
		
	float oldVal = ndx[tid.x][tid.y][tid.z];
	/*
		Heaters
	*/
	int rad = 5;
	if (vox.x - res.x / 4 > res.x / 2 - rad && vox.y > res.y / 2 - rad && vox.z > res.z / 2 - rad &&
		vox.x - res.x / 4 < res.x / 2 + rad && vox.y < res.y / 2 + rad && vox.z < res.z / 2 + rad
		)
		oldVal += 1.5f;

	if (vox.x + res.x / 4 > res.x / 2 - rad && vox.y > res.y / 2 - rad && vox.z > res.z / 2 - rad &&
		vox.x + res.x / 4 < res.x / 2 + rad && vox.y < res.y / 2 + rad && vox.z < res.z / 2 + rad
		)
		oldVal += 1.5f;

	float dt = 0.1f;

	// New heat
	float newVal = oldVal + dt * (
		ndx[tid.x - 1][tid.y][tid.z] +
		ndx[tid.x + 1][tid.y][tid.z] +
		ndx[tid.x][tid.y - 1][tid.z] +
		ndx[tid.x][tid.y + 1][tid.z] +
		ndx[tid.x][tid.y][tid.z - 1] +
		ndx[tid.x][tid.y][tid.z + 1] -
		oldVal * 6.0f
		);

	surf3Dwrite(newVal, surfOut, vox.x * sizeof(float), vox.y, vox.z);	
	
}


void launchHeatKernel(uint3 res, cudaSurfaceObject_t surfIn, cudaSurfaceObject_t surfOut) {

	const int blockSize = 8;
	const int apron = 1;


	uint3 block = make_uint3(blockSize);	
	uint3 numBlocks = make_uint3(
		(res.x / (block.x - 2 * apron)) + 1,
		(res.y / (block.y - 2 * apron)) + 1,
		(res.z / (block.z - 2 * apron)) + 1
	);

	kernelHeat<blockSize,apron><< <numBlocks, block >> > (res, surfIn, surfOut);
}







__global__ void kernelBinarizeFloat(uint3 res, cudaSurfaceObject_t surfInOut, float threshold) {

	VOLUME_VOX_GUARD(res);

	float val = 0.0f;
	surf3Dread(&val, surfInOut, vox.x * sizeof(float), vox.y, vox.z);	

	val = (val < threshold) ? 0.0f : 1.0f;	

	surf3Dwrite(val, surfInOut, vox.x * sizeof(float), vox.y, vox.z);
}

template <typename T>
__global__ void kernelBinarizeUnsigned(uint3 res, cudaSurfaceObject_t surfInOut, T threshold) {

	VOLUME_VOX_GUARD(res);

	T val = 0;
	surf3Dread(&val, surfInOut, vox.x * sizeof(T), vox.y, vox.z);

	val = (val < threshold) ? T(0) : T(-1);

	surf3Dwrite(val, surfInOut, vox.x * sizeof(T), vox.y, vox.z);
}



void launchBinarizeKernel(uint3 res, cudaSurfaceObject_t surfInOut, PrimitiveType type, float threshold) {

	uint3 block = make_uint3(8, 8, 8);
	uint3 numBlocks = make_uint3(
		(res.x / block.x) + 1,
		(res.y / block.y) + 1,
		(res.z / block.z) + 1
	);

	if (type == TYPE_FLOAT)
		kernelBinarizeFloat << <numBlocks, block >> > (res, surfInOut, threshold);
	else if (type == TYPE_UCHAR)
		kernelBinarizeUnsigned<uchar> << <numBlocks, block >> > (res, surfInOut, uchar(threshold * 255));
	else
		exit(-1);
}


template <int blockSize, int apron>
__global__ void kernelDiffuse(DiffuseParams params) {	
	static_assert(apron * 2 < blockSize, "Apron must be less than blockSize / 2");
	const uint3 res = params.res;
	const int N = blockSize;
	const int3 tid = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);

	//Sliding window of blockdim - 2*apron size
	const int3 vox = make_int3(
		blockIdx.x * (blockDim.x - 2 * apron),
		blockIdx.y * (blockDim.y - 2 * apron),
		blockIdx.z * (blockDim.z - 2 * apron)
	) + tid - make_int3(apron);

	//Toroidal boundaries	

	//Read whole block into shared memory
	__shared__ float ndx[N][N][N];
	__shared__ float Ddx[N][N][N];

	
	//Priority x > y > z (instead of 27 boundary values, just use 6)	
	Dir dir = DIR_NONE;
	if (vox.x < 0) 		
		dir = X_NEG;	
	else if (vox.x >= res.x) 
		dir = X_POS;
	else if (vox.y < 0)
		dir = Y_NEG;
	else if (vox.y >= res.y)
		dir = Y_POS;
	else if (vox.z < 0)
		dir = Z_NEG;
	else if (vox.z >= res.z)
		dir = Z_POS;
	

	if (dir != DIR_NONE) {
		ndx[tid.x][tid.y][tid.z] = params.boundaryValues[dir];
		Ddx[tid.x][tid.y][tid.z] = BOUNDARY_ZERO_GRADIENT;
	}
	else {
		surf3Dread(
			&ndx[tid.x][tid.y][tid.z],
			params.concetrationIn,
			vox.x * sizeof(float), vox.y, vox.z
		);

		uchar maskVal;
		surf3Dread(
			&maskVal,
			params.mask,
			vox.x * sizeof(uchar), vox.y, vox.z
		);
		if (maskVal == 0)
			Ddx[tid.x][tid.y][tid.z] = params.zeroDiff;
		else
			Ddx[tid.x][tid.y][tid.z] = params.oneDiff;


	}	
	__syncthreads();

	//If zero grad boundary cond, copy value from neighbour (after sync!)
	if (ndx[tid.x][tid.y][tid.z] == BOUNDARY_ZERO_GRADIENT) {		
		int3 neighVec = dirVec(dir) * -1;
		ndx[tid.x][tid.y][tid.z] = ndx[tid.x + neighVec.x][tid.y + neighVec.y][tid.z + neighVec.z];
	}

	if (Ddx[tid.x][tid.y][tid.z] == BOUNDARY_ZERO_GRADIENT) {
		int3 neighVec = dirVec(dir) * -1;
		Ddx[tid.x][tid.y][tid.z] = Ddx[tid.x + neighVec.x][tid.y + neighVec.y][tid.z + neighVec.z];
	}
	//TODO: test what is faster -> double read from global memory, or copy within shared with extra threadsync

	__syncthreads();


	//Skip apron voxels
	if (tid.x < apron || tid.x >= N - apron ||
		tid.y < apron || tid.y >= N - apron ||
		tid.z < apron || tid.z >= N - apron) return;

	//Skip outside voxels
	if (vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ||
		vox.x < 0 || vox.y < 0 || vox.z < 0)	return;

	//Load battery value
	
	uchar mask = 0;
	surf3Dread(&mask, params.mask, vox.x * sizeof(uchar), vox.y, vox.z);

	



	
	///
	{
		float dx = params.voxelSize;

		const float D = Ddx[tid.x][tid.y][tid.z];
		const float3 D3 = make_float3(D);

		const float3 Dneg = lerp(
			D3,
			make_float3(Ddx[tid.x - 1][tid.y][tid.z], Ddx[tid.x][tid.y - 1][tid.z], Ddx[tid.x][tid.y][tid.z-1]),
			(dx * 0.5f)
		);

		const float3 Dpos = lerp(
			D3,
			make_float3(Ddx[tid.x + 1][tid.y][tid.z], Ddx[tid.x][tid.y + 1][tid.z], Ddx[tid.x][tid.y][tid.z + 1]),			
			(dx * 0.5f)
		);	


		const float3 C = make_float3(ndx[tid.x][tid.y][tid.z]);

		const float3 Cneg = lerp(
			C,
			make_float3(ndx[tid.x - 1][tid.y][tid.z], ndx[tid.x][tid.y - 1][tid.z],	ndx[tid.x][tid.y][tid.z - 1]),
			dx
		);

		const float3 Cpos = lerp(
			C,
			make_float3(ndx[tid.x + 1][tid.y][tid.z], ndx[tid.x][tid.y + 1][tid.z], ndx[tid.x][tid.y][tid.z + 1]),
			dx
		);
		
		float dt = dx*dx * (1.0f / (2.0f * min(params.zeroDiff, params.oneDiff)));


		//https://math.stackexchange.com/questions/1949795/explicit-finite-difference-scheme-for-nonlinear-diffusion
		//float3 dc = (dt / (dx*dx)) * (Dpos * (Cpos - C) - Dneg * (C - Cneg));

		float3 dc = Dneg * Cneg + Dpos * Cpos - C * (Dneg + Dpos);

		//float3 dc = D * (Cpos - 2* C + Cneg) + (Dneg - Dpos) * ()

		//if (vox.x == 2 && vox.y == 10 && vox.z == 10) {
			//printf("dt: %f\n", dt);
		//}
			//printf("c: %f, D: %.9f dc: %f %f %f, Dneg: %f %f %f\n",C.x, D, dc.x, dc.y, dc.z, Dneg.x, Dneg.y, Dneg.z);

		

		float newVal = C.x + (dc.x + dc.y + dc.z);
		

		surf3Dwrite(newVal, params.concetrationOut, vox.x * sizeof(float), vox.y, vox.z);


		return;

		//float3 dD2 = make_float3(
		//	Ddx[tid.x - 1][tid.y][tid.z] + 2.0f * oldVal + Ddx[tid.x + 1][tid.y][tid.z],
		//	Ddx[tid.x][tid.y - 1][tid.z] + 2.0f * oldVal + Ddx[tid.x][tid.y + 1][tid.z],
		//	Ddx[tid.x][tid.y][tid.z - 1] + 2.0f * oldVal + Ddx[tid.x][tid.y][tid.z + 1]
		//);

		//float3 dc2 = make_float3(
		//	ndx[tid.x - 1][tid.y][tid.z] + 2.0f * oldVal + ndx[tid.x + 1][tid.y][tid.z],
		//	ndx[tid.x][tid.y - 1][tid.z] + 2.0f * oldVal + ndx[tid.x][tid.y + 1][tid.z],
		//	ndx[tid.x][tid.y][tid.z - 1] + 2.0f * oldVal + ndx[tid.x][tid.y][tid.z + 1]
		//);


		

	}

	
}

void launchDiffuseKernel(DiffuseParams params) {

	const int blockSize = 8;
	const int apron = 1;

	uint3 res = params.res;

	uint3 block = make_uint3(blockSize);
	uint3 numBlocks = make_uint3(
		(res.x / (block.x - 2 * apron)) + 1,
		(res.y / (block.y - 2 * apron)) + 1,
		(res.z / (block.z - 2 * apron)) + 1
	);

	kernelDiffuse<blockSize, apron> << <numBlocks, block >> > (params);
}







__global__ void kernelSubtract(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B) {

	VOLUME_VOX_GUARD(res);

	float Aval, Bval;
	surf3Dread(&Aval, A, vox.x * sizeof(float), vox.y, vox.z);
	surf3Dread(&Bval, B, vox.x * sizeof(float), vox.y, vox.z);

	float newVal = Bval - Aval;
	
	surf3Dwrite(newVal, A, vox.x * sizeof(float), vox.y, vox.z);
}

void launchSubtractKernel(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B) {
	uint3 block = make_uint3(8, 8, 8);
	uint3 numBlocks = make_uint3(
		(res.x / block.x) + 1,
		(res.y / block.y) + 1,
		(res.z / block.z) + 1
	);

	kernelSubtract << <numBlocks, block >> > (res, A, B);

}



//template <typename T, unsigned int blockSize>
//__global__ void reduce(T *g_idata, T *g_odata, unsigned int n)
//{
//	extern __shared__ int sdata[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//	sdata[tid] = 0;
//
//	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
//	__syncthreads();
//
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//	if (tid < 32) {
//		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}

__host__ __device__ uint3 ind2sub(uint3 res, uint i) {
	uint x = i % res.x;
	uint tmp = ((i - x) / res.x);
	uint y = tmp % res.y;
	uint z = (tmp - y) / res.y;
	return make_uint3(
		x,y,z
	);
}


template <typename T, unsigned int blockSize, bool toSurface>
__global__ void reduce3D(uint3 res, cudaSurfaceObject_t data, T * finalData, unsigned int n, uint3 offset)
{
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) { 
		uint3 voxi = ind2sub(res, i);
		uint3 voxip = ind2sub(res, i + blockSize); 

		T vali, valip = 0;
		surf3Dread(&vali, data, voxi.x * sizeof(T), voxi.y, voxi.z);
		if(voxip.x < res.x && voxip.y < res.y && voxip.z < res.z)
			surf3Dread(&valip, data, voxip.x * sizeof(T), voxip.y, voxip.z);	
				
	
		sdata[tid] += (vali + valip);

		i += gridSize; 
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) {
		unsigned int o = blockIdx.x;	
		//Either copy to surface
		if (toSurface) {						
			uint3 voxo = ind2sub(res,o);
			surf3Dwrite(sdata[0], data, voxo.x * sizeof(T), voxo.y, voxo.z);				
		}
		//Or final 1D array
		else {
			finalData[o] = sdata[0];						
		}	
	}
	
}



float launchReduceSumKernel(uint3 res, cudaSurfaceObject_t surf) {
		

	const uint finalSizeMax = 512;	
	const uint blockSize = 512;
	const uint3 block = make_uint3(blockSize,1,1);	
	uint n = res.x * res.y * res.z;

	//uint finalSizeMax = ((res.x * res.y * res.z) / blockSize) / 2;


	float * deviceResult = nullptr;
	cudaMalloc(&deviceResult, finalSizeMax * sizeof(float));
	cudaMemset(deviceResult, 0, finalSizeMax * sizeof(float));
	

	while (n > finalSizeMax) {
		uint3 numBlocks = make_uint3(
			(n / block.x) / 2 , 1, 1
		);		

		//If not final stage of reduction -> reduce into  surface
		if (numBlocks.x > finalSizeMax) {
			reduce3D<float, blockSize, true>
				<<<numBlocks, block, blockSize * sizeof(float)>>> (
					res, surf, nullptr, n, make_uint3(0)
					);
		}
		else {
			reduce3D<float, blockSize, false>
				<<<numBlocks, block, blockSize * sizeof(float)>>> (
					res, surf, deviceResult, n, make_uint3(0)
					);
		
		}

		//New N
		n = numBlocks.x;
	}


	float * hostResult = new float[finalSizeMax];
	cudaMemcpy(hostResult, deviceResult, finalSizeMax * sizeof(float), cudaMemcpyDeviceToHost);


	float result = 0.0f;
	for (auto i = 0; i < finalSizeMax; i++) {
		result += hostResult[i];
	}

	cudaFree(deviceResult);
	delete[] hostResult;


	return result;

}











/*
	Surface & buffer reduction, templated	
*/


template <typename T>
__device__ void opSum(volatile T & a, T b) {
	a += b;
}

template <typename T>
__device__ void opProd(volatile T & a, T b) {
	a *= b;
}

template <typename T>
__device__ void opMin(volatile T & a, T b) {
	if (b < a) a = b;
}
template <typename T>
__device__ void opMax(volatile T & a, T  b) {
	if (b > a) a = b;
}



template <typename T>
using ReduceOp = void(*)(
	volatile T & a, T b
	);



template <typename T, typename R = T>
__device__ R opSquare(T & a) {
	return R(a)*R(a);
}

template <typename T, typename R = T>
__device__ R opZeroElem(T & a) {
	return (a == T(0)) ? R(1) : R(0);
}

template <typename T, typename R = T>
__device__ R opIdentity(T & a) {
	return R(a);
}

template <typename T, typename R = T>
using PreReduceOp = R(*)(
	T & a
	);

template <typename T, typename R, unsigned int blockSize, ReduceOp<R> _op, PreReduceOp<T,R> _preOp = opIdentity<T>>
__global__ void reduce3DSurfaceToBufferTR(
	uint3 res, 
	cudaSurfaceObject_t surf, 
	R * reducedData, 
	size_t n,
	uint3 begin,
	uint3 end
)
{
	extern __shared__ __align__(sizeof(R)) volatile unsigned char my_smem[];
	volatile R *sdata = reinterpret_cast<volatile R *>(my_smem);

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = R(0);

	while (i < n) {
		const uint3 voxi = ind2sub(res, i);
		const uint3 voxip = ind2sub(res, i + blockSize);		
		if (_isValidPosInRange(begin,end,voxi)) {
			T vali = read<T>(surf, voxi);			
			R valiprep = _preOp(vali); 
			_op(sdata[tid], valiprep);
		}

		if (i + blockSize < n && _isValidPosInRange(begin, end, voxi)) {
			T valip = read<T>(surf, voxip);			
			R valiprep = _preOp(valip); 
			_op(sdata[tid], valiprep);
		}

		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { _op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { _op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { _op(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) _op(sdata[tid], sdata[tid + 32]);
		if (blockSize >= 32) _op(sdata[tid], sdata[tid + 16]);
		if (blockSize >= 16) _op(sdata[tid], sdata[tid + 8]);
		if (blockSize >= 8) _op(sdata[tid], sdata[tid + 4]);
		if (blockSize >= 4) _op(sdata[tid], sdata[tid + 2]);
		if (blockSize >= 2) _op(sdata[tid], sdata[tid + 1]);
	}

	if (tid == 0) {
		reducedData[blockIdx.x] = sdata[0];
	}

}


template <typename T, unsigned int blockSize, ReduceOp<T> _op, PreReduceOp<T> _preOp = opIdentity<T>>
__global__ void reduce3DSurfaceToBuffer(uint3 res, cudaSurfaceObject_t surf, T * reducedData, size_t n)
{
	extern __shared__ __align__(sizeof(T)) volatile unsigned char my_smem[];
	volatile T *sdata = reinterpret_cast<volatile T *>(my_smem);

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = T(0);

	while (i < n) {
		const uint3 voxi = ind2sub(res, i);
		const uint3 voxip = ind2sub(res, i + blockSize);

		if (voxi.x < res.x && voxi.y < res.y && voxi.z < res.z) {
			
			T vali = T(0);
			vali = read<T>(surf, voxi);
			T valiprep = _preOp(vali); //todo
			_op(sdata[tid], valiprep);
		}

		if (i + blockSize < n && voxip.x < res.x && voxip.y < res.y && voxip.z < res.z) {
			T valip = T(0);
			valip = read<T>(surf, voxip);
			T valiprep = _preOp(valip); //todo
			_op(sdata[tid], valiprep);
		}		

		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { _op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { _op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { _op(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) _op(sdata[tid], sdata[tid + 32]);
		if (blockSize >= 32) _op(sdata[tid], sdata[tid + 16]);
		if (blockSize >= 16) _op(sdata[tid], sdata[tid + 8]);
		if (blockSize >= 8) _op(sdata[tid], sdata[tid + 4]);
		if (blockSize >= 4) _op(sdata[tid], sdata[tid + 2]);
		if (blockSize >= 2) _op(sdata[tid], sdata[tid + 1]);
	}

	if (tid == 0) {		
		reducedData[blockIdx.x] = sdata[0];
	}

}


template <typename T, unsigned int blockSize, ReduceOp<T> _op>
__global__ void reduceBuffer(T * buffer, size_t n)
{
	extern __shared__ __align__(sizeof(T)) volatile unsigned char my_smem[];
	volatile T *sdata = reinterpret_cast<volatile T *>(my_smem);
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n) {
		_op(sdata[tid], buffer[i]);

		if(i + blockSize < n)
			_op(sdata[tid], buffer[i + blockSize]);

		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { _op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { _op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { _op(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) _op(sdata[tid], sdata[tid + 32]);
		if (blockSize >= 32) _op(sdata[tid], sdata[tid + 16]);
		if (blockSize >= 16) _op(sdata[tid], sdata[tid + 8]);
		if (blockSize >= 8) _op(sdata[tid], sdata[tid + 4]);
		if (blockSize >= 4) _op(sdata[tid], sdata[tid + 2]);
		if (blockSize >= 2) _op(sdata[tid], sdata[tid + 1]);
	}

	if (tid == 0) {
		buffer[blockIdx.x] = sdata[0];
	}

}



template<typename In, typename Out, ReduceOp<Out> _op, PreReduceOp<In, Out> _preOp>
void _Volume_Reduce_Op(CUDA_Volume & vol, ReduceOpType opType, PrimitiveType outputType, void * auxBufferGPU, void * auxBufferCPU, void * result,
	uint3 begin,
	uint3 end) {


	begin = make_uint3(min(begin.x, vol.res.x), min(begin.y, vol.res.y), min(begin.z, vol.res.z));
	end = make_uint3(min(end.x, vol.res.x), min(end.y, vol.res.y), min(end.z, vol.res.z));

	//printf("Reduction from %d %d %d to %u %u %u\n", begin.x, begin.y, begin.z, end.x, end.y, end.z);
	
	

	const uint blockSize = VOLUME_REDUCTION_BLOCKSIZE;
	const uint sharedSize = primitiveSizeof(outputType) * blockSize;

	const uint3 block = make_uint3(blockSize, 1, 1);
	const uint finalSizeMax = VOLUME_REDUCTION_BLOCKSIZE;


	const size_t initialN = vol.res.x * vol.res.y * vol.res.z;
	size_t n = initialN;

	/*
	Reduce from surface to auxiliar buffer
	*/
	{
		uint3 numBlocks = make_uint3(
			//uint((n / block.x) / 2)
			roundDiv(roundDiv(uint(n), block.x), 2)
			, 1, 1
		);
		if (numBlocks.x == 0)
			numBlocks.x = 1;

		reduce3DSurfaceToBufferTR<In,Out, blockSize, _op, _preOp> << <numBlocks, block, sharedSize >>> (
			vol.res, vol.surf, (Out*)auxBufferGPU, n, begin, end
			);
		n = numBlocks.x;
	}

	/*
	Further reduce in buffer
	*/
	while (n > finalSizeMax) {
		const uint blockSize = VOLUME_REDUCTION_BLOCKSIZE;
		const uint3 block = make_uint3(blockSize, 1, 1);
		uint3 numBlocks = make_uint3(
			roundDiv(roundDiv(uint(n),block.x),2),//(n / block.x) / 2)			
			 1, 1
		);
		

		reduceBuffer<Out, blockSize, _op> << <numBlocks, block, sharedSize >> > ((Out*)auxBufferGPU, n);
		n = numBlocks.x;
	}

	cudaMemcpy(auxBufferCPU, auxBufferGPU, primitiveSizeof(outputType) * n, cudaMemcpyDeviceToHost);

	/*
	Sum last array on CPU
	*/
	*((Out*)result) = Out(0);
	for (auto i = 0; i < n; i++) {
		*((Out*)result) += ((Out*)auxBufferCPU)[i];
	}

	return;
}


template<typename In, typename Out>
void _Volume_Reduce_Out(CUDA_Volume & vol, ReduceOpType opType, PrimitiveType outputType, void * auxBufferGPU, void * auxBufferCPU, void * result,
	uint3 begin,
	uint3 end) {

	switch (opType) {
	case REDUCE_OP_MIN:
		return _Volume_Reduce_Op<In, Out, opMin<Out>, opIdentity<In,Out>>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case REDUCE_OP_MAX:
		return _Volume_Reduce_Op<In, Out, opMax<Out>, opIdentity<In, Out>>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case REDUCE_OP_SUM:
		return _Volume_Reduce_Op<In, Out, opSum<Out>, opIdentity<In, Out>>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case REDUCE_OP_SQUARESUM:
		return _Volume_Reduce_Op<In, Out, opSum<Out>, opSquare<In, Out>>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case REDUCE_OP_PROD:
		return _Volume_Reduce_Op<In, Out, opProd<Out>, opIdentity<In, Out>>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case REDUCE_OP_SUM_ZEROELEM:
		return _Volume_Reduce_Op<In, Out, opSum<Out>, opZeroElem<In, Out>>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	default:
		assert("Unsupported reduction op");
		exit(0);
	}

}

template<typename In>
void _Volume_Reduce_In(CUDA_Volume & vol, ReduceOpType opType, PrimitiveType outputType, void * auxBufferGPU, void * auxBufferCPU, void * result,
	uint3 begin,
	uint3 end) {

	switch (outputType) {
	case TYPE_FLOAT:
		return _Volume_Reduce_Out<In, float>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_DOUBLE:
		return _Volume_Reduce_Out<In, double>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_CHAR:
		return _Volume_Reduce_Out<In, char>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_UCHAR:
		return _Volume_Reduce_Out<In, uchar>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_UINT:
		return _Volume_Reduce_Out<In, uint>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_UINT64:
		return _Volume_Reduce_Out<In, uint64>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	default:
		assert("Unsupported type");
		exit(0);
	}	
}

void Volume_Reduce(CUDA_Volume & vol, ReduceOpType opType, PrimitiveType outputType, void * auxBufferGPU, void * auxBufferCPU, void * result,
	uint3 begin, 
	uint3 end
)
{

	

	if ((vol.type == TYPE_CHAR || vol.type == TYPE_UCHAR) &&
		(outputType == TYPE_CHAR || outputType == TYPE_UCHAR) &&
		opType != REDUCE_OP_MIN && opType != REDUCE_OP_MAX
		) {
		printf("Warning: Reduction can result in overflow. Choose wider output type.\n");
	}

	switch (vol.type) {
	case TYPE_FLOAT:
		return _Volume_Reduce_In<float>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_DOUBLE:
		return _Volume_Reduce_In<double>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_CHAR:
		return _Volume_Reduce_In<char>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	case TYPE_UCHAR:
		return _Volume_Reduce_In<uchar>(vol, opType, outputType, auxBufferGPU, auxBufferCPU, result, begin, end);
	default:
		assert("Unsupported type");
		exit(0);
	}
}




//AuxBuffer -> surf total n / 512
/*
void launchReduceKernel(	
	PrimitiveType type, 
	ReduceOpType opType,
	uint3 res, 
	cudaSurfaceObject_t surf, 
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result	
) {
	
	
	

	const uint blockSize = VOLUME_REDUCTION_BLOCKSIZE;
	const uint sharedSize = primitiveSizeof(type) * blockSize;
	
	const uint3 block = make_uint3(blockSize, 1, 1);
	const uint finalSizeMax = VOLUME_REDUCTION_BLOCKSIZE;
	const size_t initialN = res.x * res.y * res.z;
	
	size_t n = initialN;
		
	/ *
		Reduce from surface to auxiliar buffer
	* /
	{		
		uint3 numBlocks = make_uint3(
			uint((n / block.x) / 2), 1, 1
		);
		if (numBlocks.x == 0)
			numBlocks.x = 1;

		if (type == TYPE_FLOAT) {
			if (opType == REDUCE_OP_SQUARESUM)
				reduce3DSurfaceToBuffer<float, blockSize, opSum, opSquare<float,float>> << <numBlocks, block, sharedSize >> > (
					res, surf, (float*)auxBufferGPU, n
					);

			else if (opType == REDUCE_OP_MIN)
				reduce3DSurfaceToBuffer<float, blockSize, opMin, opIdentity> << <numBlocks, block, sharedSize >> > (
					res, surf, (float*)auxBufferGPU, n
					);
			else if (opType == REDUCE_OP_MAX)
				reduce3DSurfaceToBuffer<float, blockSize, opMax, opIdentity> << <numBlocks, block, sharedSize >> > (
					res, surf, (float*)auxBufferGPU, n
					);
			else if (opType == REDUCE_OP_SUM)
				reduce3DSurfaceToBuffer<float, blockSize, opSum, opIdentity> << <numBlocks, block, sharedSize >> > (
					res, surf, (float*)auxBufferGPU, n
					);
			else
				exit(0);
			
		}
		else if (type == TYPE_DOUBLE) {
			if (opType == REDUCE_OP_SQUARESUM) {
				reduce3DSurfaceToBuffer<double, blockSize, opSum, opSquare> << <numBlocks, block, sharedSize >> > (
					res, surf, (double*)auxBufferGPU, n
					);
			}		
			else if (opType == REDUCE_OP_MIN)
				reduce3DSurfaceToBuffer<double, blockSize, opMin, opIdentity> << <numBlocks, block, sharedSize >> > (
					res, surf, (double*)auxBufferGPU, n
					);
			else if (opType == REDUCE_OP_MAX)
				reduce3DSurfaceToBuffer<double, blockSize, opMax, opIdentity> << <numBlocks, block, sharedSize >> > (
					res, surf, (double*)auxBufferGPU, n
					);
			else if (opType == REDUCE_OP_SUM)
				reduce3DSurfaceToBuffer<double, blockSize, opSum, opIdentity> << <numBlocks, block, sharedSize >> > (
					res, surf, (double*)auxBufferGPU, n
					);
			else
				exit(0);
			
		}

		n = numBlocks.x;
	}


	/ *
		Further reduce in buffer
	* /
	while (n > finalSizeMax) {
		const uint blockSize = VOLUME_REDUCTION_BLOCKSIZE;
		const uint3 block = make_uint3(blockSize, 1, 1);
		uint3 numBlocks = make_uint3(
				uint((n / block.x) / 2), 1, 1
			);

		if (type == TYPE_FLOAT) {
			if (opType == REDUCE_OP_SQUARESUM || opType == REDUCE_OP_SUM)
				reduceBuffer<float, blockSize, opSum> << <numBlocks, block, sharedSize >> > ((float*)auxBufferGPU, n);
			else if (opType == REDUCE_OP_MIN)
				reduceBuffer<float, blockSize, opMin> << <numBlocks, block, sharedSize >> > ((float*)auxBufferGPU, n);
			else if (opType == REDUCE_OP_MAX)
				reduceBuffer<float, blockSize, opMax> << <numBlocks, block, sharedSize >> > ((float*)auxBufferGPU, n);			
			else
				exit(0);
		}
		if (type == TYPE_DOUBLE) {
			if (opType == REDUCE_OP_SQUARESUM || opType == REDUCE_OP_SUM)
				reduceBuffer<double, blockSize, opSum> << <numBlocks, block, sharedSize >> >((double*)auxBufferGPU, n);
			else if (opType == REDUCE_OP_MIN)
				reduceBuffer<double, blockSize, opMin> << <numBlocks, block, sharedSize >> >((double*)auxBufferGPU, n);
			else if (opType == REDUCE_OP_MAX)
				reduceBuffer<double, blockSize, opMax> << <numBlocks, block, sharedSize >> >((double*)auxBufferGPU, n);			
			else
				exit(0);
		}


		n = numBlocks.x;
	}
	

	
	cudaMemcpy(auxBufferCPU, auxBufferGPU, primitiveSizeof(type) * n, cudaMemcpyDeviceToHost);

	/ *
		Sum last array on CPU
	* /
	if (type == TYPE_FLOAT) {	
		*((float*)result) = 0.0f;
		
		for (auto i = 0; i < n; i++) {
			//printf("%f\n", ((float*)auxBufferCPU)[i]);			
			*((float*)result) += ((float*)auxBufferCPU)[i];
		}
	}
	else if (type == TYPE_DOUBLE) {	
		*((double*)result) = 0.0;
		for (auto i = 0; i < n; i++) {
			*((double*)result) += ((double*)auxBufferCPU)[i];
		}		
	}

	
	

	return;



}*/



template <typename T>
__global__ void __clearKernel(cudaSurfaceObject_t surf, uint3 res, T val) {
	VOLUME_VOX_GUARD(res);
	write<T>(surf, vox, val);
}

void launchClearKernel(PrimitiveType type, cudaSurfaceObject_t surf, uint3 res, void * val) {
	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		__clearKernel<float> << < numBlocks, block >> >(surf, res, *((float *)val));
	}
	else if (type == TYPE_DOUBLE)
		__clearKernel<double> << < numBlocks, block >> >(surf, res, *((double *)val));
}


template <typename T>
__global__ void __normalizeKernel(cudaSurfaceObject_t surf, uint3 res, T low, T high) {
	VOLUME_VOX_GUARD(res);
	T oldval = read<T>(surf, vox);
	write<T>(surf, vox, (oldval - low) / (high - low));
}



void launchNormalizeKernel(
	PrimitiveType type, cudaSurfaceObject_t surf, uint3 res, double low, double high
) {
	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		__normalizeKernel<float> << < numBlocks, block >> >(surf, res,float(low), float(high));
	}
	else if (type == TYPE_DOUBLE)
		__normalizeKernel<double> << < numBlocks, block >> >(surf, res,low, high);

}

/////////////////////////////////////////////

template <typename T>
__global__ void __copyKernel(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B) {
	VOLUME_VOX_GUARD(res)
		write<T>(B, vox, read<T>(A, vox));
}


void launchCopyKernel(PrimitiveType type, uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B) {
	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		__copyKernel<float> << < numBlocks, block >> >(res, A, B);
	}
	else if (type == TYPE_DOUBLE)
		__copyKernel<double> << < numBlocks, block >> >(res, A, B);
}




template <typename T>
__global__ void __multiplyKernel(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B, cudaSurfaceObject_t C) {
	VOLUME_VOX_GUARD(res)
		write<T>(C, vox,
			read<T>(A, vox) * read<T>(B, vox)
			);
}

void launchMultiplyKernel(PrimitiveType type, uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B, cudaSurfaceObject_t C) {
	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		__multiplyKernel<float> << < numBlocks, block >> >(res, A, B, C);
	}
	else if (type == TYPE_DOUBLE)
		__multiplyKernel<double> << < numBlocks, block >> >(res, A, B, C);
}


void Volume_DotProduct(
	CUDA_Volume A, 
	CUDA_Volume B,
	CUDA_Volume C,
	void * auxBufferGPU,
	void * auxBufferCPU,
	void * result	
) {
	assert(A.res.x == B.res.x && A.res.y == B.res.y && A.res.z == B.res.z && A.type == B.type);
	assert(A.res.x == C.res.x && A.res.y == C.res.y && A.res.z == C.res.z && A.type == C.type);
	launchMultiplyKernel(A.type, A.res, A.surf, B.surf, C.surf);
	Volume_Reduce(C, REDUCE_OP_SUM, C.type, auxBufferGPU, auxBufferCPU, result);	
}





template <typename T>
__global__ void __addAPlusBetaBKernel(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B, cudaSurfaceObject_t C, T beta) {
	VOLUME_VOX_GUARD(res)		
		write<T>(C, vox,
			read<T>(A, vox) + beta  * read<T>(B, vox)
			);
}




void launchAddAPlusBetaB(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C,
	double beta
) {

	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		__addAPlusBetaBKernel<float> << < numBlocks, block >> >(res, A, B, C, float(beta));
	}
	else if (type == TYPE_DOUBLE)
		__addAPlusBetaBKernel<double> << < numBlocks, block >> >(res, A, B, C, beta);
}




template <typename T>
__global__ void __aPlusBetaBGammaPlusC(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B, cudaSurfaceObject_t C, T beta, T gamma) {
	VOLUME_VOX_GUARD(res)

		//A = gamma * (A + beta * B) + C
		write<T>(A, vox,
			gamma * (read<T>(A, vox) + beta  * read<T>(B, vox)) + read<T>(C, vox)
			);
}


void launchAPlusBetaBGammaPlusC(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C,
	double beta,
	double gamma
) {

	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		__aPlusBetaBGammaPlusC<float> << < numBlocks, block >> >(res, A, B, C, float(beta), float(gamma));
	}
	else if (type == TYPE_DOUBLE)
		__aPlusBetaBGammaPlusC<double> << < numBlocks, block >> >(res, A, B, C, beta, gamma);
}



template <typename T>
__global__ void ___ABCBetaGamma(uint3 res, cudaSurfaceObject_t A, cudaSurfaceObject_t B, cudaSurfaceObject_t C, T beta, T gamma) {
	VOLUME_VOX_GUARD(res)

		//A = A + beta * B + gamma * C
		write<T>(A, vox,
			read<T>(A, vox) + beta  * read<T>(B, vox) + gamma * read<T>(C, vox)
			);
}

//A = A + beta * B + gamma * C
void launchABC_BetaGamma(
	PrimitiveType type,
	uint3 res,
	cudaSurfaceObject_t A,
	cudaSurfaceObject_t B,
	cudaSurfaceObject_t C,
	double beta,
	double gamma
) {

	BLOCKS3D(8, res);
	if (type == TYPE_FLOAT) {
		___ABCBetaGamma<float> << < numBlocks, block >> >(res, A, B, C, float(beta), float(gamma));
	}
	else if (type == TYPE_DOUBLE)
		___ABCBetaGamma<double> << < numBlocks, block >> >(res, A, B, C, beta, gamma);
}



double Volume_SquareNorm(
	const uint3 res,
	CUDA_Volume & x,
	void * auxGPU,
	void * auxCPU
) {

	double result = 0.0;

	Volume_Reduce(x, REDUCE_OP_SQUARESUM, x.type, auxGPU, auxCPU, &result);
	/*launchReduceKernel(
		TYPE_DOUBLE,
		REDUCE_OP_SQUARESUM,
		res,
		x.surf,
		auxGPU,
		auxCPU,
		&result
	);
	*/
	return result;

}

void Volume_SetToZero(
	CUDA_Volume & x
) {
	double val = 0.0;
	launchClearKernel(TYPE_DOUBLE, x.surf, x.res, &val);
}
