#pragma once

#include "cuda/ManagedAllocator.cuh"

namespace fast {

	class ResourceImpl {		
	public:
		void * ptr();
		const void * ptr() const;
		void resize(size_t N);
		void clear();
	private:
		managed_device_vector<unsigned char> _data;
	};

	

}