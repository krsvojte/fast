#pragma once

#include <fastlib/solver/LinearSystem.h>
#include <fastlib/cuda/ManagedAllocator.cuh>

namespace fast {

	template <typename T>
	struct LinearSys_StencilDevicePtr {
		T * dir[7];
	};

	template <typename T>
	struct LinearSys_Stencil {
		managed_device_vector<T> dir[7];
		LinearSys_StencilDevicePtr<T> getPtr() {
			return {
				thrust::raw_pointer_cast(&dir[0].front()),
				thrust::raw_pointer_cast(&dir[1].front()),
				thrust::raw_pointer_cast(&dir[2].front()),
				thrust::raw_pointer_cast(&dir[3].front()),
				thrust::raw_pointer_cast(&dir[4].front()),
				thrust::raw_pointer_cast(&dir[5].front()),
				thrust::raw_pointer_cast(&dir[6].front())
			};
		}
	};

	template <typename T>
	class LinearSystemDevice : public LinearSystem {
	public:
		LinearSys_Stencil<T> A;
		managed_device_vector<T> x;
		managed_device_vector<T> b;
	
		virtual PrimitiveType type() override
		{
			return primitiveTypeof<T>();
		}

	};

}