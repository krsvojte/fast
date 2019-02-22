#include "cuda/ResourceImpl.cuh"


void * fast::ResourceImpl::ptr() {
	return thrust::raw_pointer_cast(&_data.front());
}

const void * fast::ResourceImpl::ptr() const {
	return thrust::raw_pointer_cast(&_data.front());
}


void fast::ResourceImpl::resize(size_t N)
{
	_data.resize(N);
}

void fast::ResourceImpl::clear()
{
	_data.clear();
}
