/*
#include "utility/Resource.h"


#include "ResourceImpl.cuh"


fast::Resource::Resource(PrimitiveType type, int dim) :
	_impl(
		std::make_unique<ResourceImpl>()
	), _x(0),_y(0),_z(0)
{

}

bool fast::Resource::resize(int x, int y / *= 0* /, int z / *= 0* /)
{
	if (getDimension() == 1) {
		if (y || z) return false;
		_impl->resize(x);
#ifdef UNIFORM_MEMORY_DISABLED
		if()
		_cpuData.resize(x);
#endif
		_x = x;
	}
		

	if (getDimension() == 2) {
		if (z) return false;
		_impl->resize(x*y);
		_x = x;
		_y = y;
	}

	if (getDimension() == 3) {		
		_impl->resize(x*y*z);
		_x = x;
		_y = y;
		_z = z;
	}
}

int fast::Resource::getDimension()
{
	return _dim;
}

PrimitiveType fast::Resource::getType() const
{
	return _type;
}

fast::Resource::Array1D fast::Resource::getArray1D(Location loc) const
{
	assert(_dim == 1);

#ifdef UNIFORM_MEMORY_DISABLED
	if(loc == LOCATION_HOST)
		return {
			_cpuData.data(),
			getType(),
			_x
		};
#endif

	return {
		_impl->ptr(),
		getType(),
		_x
	};

		
	
}

fast::Resource::Array2D fast::Resource::getArray2D(Location loc) const
{
	assert(_dim == 2);
#ifdef UNIFORM_MEMORY_DISABLED
	if (loc == LOCATION_HOST)
		return {
		_cpuData.data(),
		getType(),
		_x, _y
	};
#endif

	return {
		_impl->ptr(),
		getType(),
		_x, _y
	};
}

fast::Resource::Array3D fast::Resource::getArray3D(Location loc) const
{
	assert(_dim == 3);

#ifdef UNIFORM_MEMORY_DISABLED
	if (loc == LOCATION_HOST)
		return {
		_cpuData.data(),
		getType(),
		_x, _y, _z
	};
#endif
	return {
		_impl->ptr(),
		getType(),
		_x, _y, _z
	};
}
*/
