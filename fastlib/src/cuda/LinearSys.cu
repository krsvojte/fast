
#include "LinearSys.cuh"

#include <memory>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

template <typename T>
void copyDeviceToHost(const managed_device_vector<T> & device, T * host)
{
	cudaMemcpy(host, THRUST_PTR(device), sizeof(T)*device.size(), cudaMemcpyDeviceToHost);
}

template <typename T>
void copyHostToDevice(const T* host, managed_device_vector<T> & device)
{
	cudaMemcpy(THRUST_PTR(device), host, sizeof(T)*device.size(), cudaMemcpyHostToDevice);
}



template void copyDeviceToHost(const managed_device_vector<double> & device, double * host);
template void copyDeviceToHost(const managed_device_vector<float> & device, float * host);
template void copyHostToDevice(const double* host, managed_device_vector<double> & device);
template void copyHostToDevice(const float* host, managed_device_vector<float> & device);


template <typename T>
__global__ void __copySliceKernel(
	uint3 resSrc,
	uint3 res,
	uint3 dirs,
	int sgn,
	const T * x,
	T * tmp
) {
	VOLUME_VOX_GUARD(res); //Vox is sec0,sec1,prim

						   //_at<uint>(vox, 2) = (sgn < 0) ? 1 : (_at<uint>(resSrc, 2) - 1);

						   //Convert to x,y,z in x
	uint3 srcVox = make_uint3(0);
	_at<uint>(srcVox, dirs.x) = vox.x;
	_at<uint>(srcVox, dirs.y) = vox.y;
	_at<uint>(srcVox, dirs.z) = (sgn < 0) ? 0 : _at<uint>(resSrc, dirs.z) - 1;


	size_t srcIndex = _linearIndex(resSrc, srcVox);
	size_t targetIndex = vox.x + vox.y*res.x;

	
	tmp[targetIndex] = x[srcIndex];
}

template <typename T>
bool copySlice(const managed_device_vector<T> & x, uint3 res, Dir dir, T* output)
{

	const int primaryDir = getDirIndex(dir);
	const int sgn = getDirSgn(dir);
	const int secondaryDirs[2] = { (primaryDir + 1) % 3, (primaryDir + 2) % 3 };
	uint3 dirs = make_uint3(secondaryDirs[0], secondaryDirs[1], primaryDir);

	size_t N = _at<uint>(res, secondaryDirs[0]) * _at<uint>(res, secondaryDirs[1]);
	managed_device_vector<T> tmp;
	tmp.resize(N, T(0.0));
	{
		uint3 kernelRes = make_uint3(
			_at<uint>(res, secondaryDirs[0]),
			_at<uint>(res, secondaryDirs[1]),
			1
		);
		BLOCKS3D(8, kernelRes);

		__copySliceKernel<T> << <numBlocks, block >> > (res, kernelRes, dirs, sgn, THRUST_PTR(x), THRUST_PTR(tmp));
	}
	cudaMemcpy(output, THRUST_PTR(tmp), N * sizeof(T), cudaMemcpyDeviceToHost);
	return true;
}


template<typename T, typename K>
__global__ void vectorToVolume(CUDA_Volume volume, const T * vec) {
	VOLUME_IVOX_GUARD(volume.res);
	size_t I = _linearIndex(volume.res, ivox);

	T value = 0;

	
	value = vec[I];
	

	write<K>(volume.surf, ivox, K(value));

}

template <typename T>
void copyVectorToVolume(const managed_device_vector<T> & x, CUDA_Volume & out)
{
	BLOCKS3D(8, out.res);
	
	if (out.type == TYPE_DOUBLE) {
		vectorToVolume<T, double> << < numBlocks, block >> > (out, THRUST_PTR(x));
	}
	if (out.type == TYPE_FLOAT) {
		vectorToVolume<T, float> << < numBlocks, block >> > (out, THRUST_PTR(x));
	}
}


template bool copySlice(const managed_device_vector<double> & x, uint3 res, Dir d, double* output);
template bool copySlice(const managed_device_vector<float> & x, uint3 res, Dir d, float* output);


template void copyVectorToVolume(const managed_device_vector<double> & x, CUDA_Volume & out);
template void copyVectorToVolume(const managed_device_vector<float> & x, CUDA_Volume & out);

