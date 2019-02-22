
#pragma once
#include "Volume.cuh"
#include "cuda/ManagedAllocator.cuh"

template <typename T>
void copyDeviceToHost(const managed_device_vector<T> & device, T* host);

template <typename T>
void copyHostToDevice(const T* host, managed_device_vector<T> & device);

template <typename T>
bool copySlice(const managed_device_vector<T> & x, uint3 res, Dir d, T* output);


template <typename T>
void copyVectorToVolume(
	const managed_device_vector<T> & x,
	CUDA_Volume & out
);
