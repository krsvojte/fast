#include "volume/Volume.h"

#include "cuda/VolumeTypes.cuh"
#include "cuda/Volume.cuh"
#include "cuda/CudaUtility.h"

#include "utility/GLGlobal.h"

#include <cstring>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>



using namespace fast;



fast::Volume::Volume(ivec3 dim, PrimitiveType type)
	: _dim(dim), _type(type)
{	

	if (dim.x == 0 || dim.y == 0 || dim.z == 0)
		throw "Volume: Invalid dimension (0)";

	//Allocate buffer(s)
	if (Volume::enableOpenGLInterop) {
		initGLEW();

		_ptr.allocOpenGL(type, dim, true);		
	}
	else {
		_ptr.alloc(type, dim, true);		
	}
	

	
	//Test fill
	auto & ptr = getPtr();	
	float * arr = reinterpret_cast<float*>(ptr.getCPU());	
	memset(arr, 0, ptr.byteSize());
	ptr.commit();
}



FAST_EXPORT Volume & fast::Volume::operator=(Volume && other)
{
	if (this == &other) return *this;

	other._cudaVolumeImpl = std::move(other._cudaVolumeImpl);
	_type = other._type;
	_dim = other._dim;
	_ptr = std::move(other._ptr);
	return *this;
}

fast::Volume::Volume(Volume && other)
{
	other._cudaVolumeImpl = std::move(other._cudaVolumeImpl);
	_type = other._type;
	_dim = other._dim;
	_ptr = std::move(other._ptr);	
}

fast::Volume::~Volume() {}

/*
fast::Volume::~Volume()
{

}*/

fast::Volume::Volume(Texture3DPtr && ptr)
	: _ptr(std::move(ptr)), _dim(ptr.dim()), _type(ptr.type())
{

	if (_dim.x == 0 || _dim.y == 0 || _dim.z == 0)
		throw "Volume: Invalid dimension (0)";

}

fast::Texture3DPtr & fast::Volume::getPtr()
{
	return _ptr;
}





void fast::Volume::resize(ivec3 origin, ivec3 newDim)
{
	 
	/* const ivec3 end = origin + newDim;
	 assert(end.x <= _dim.x);
	 assert(end.y <= _dim.y);
	 assert(end.z <= _dim.z);
 

	 Volume tmp = Volume(newDim, _type);


	 const auto & p0 = _ptr;
	 const auto stride = p0.stride();
	 const uchar * data0 = reinterpret_cast<const uchar *>(p0.getCPU());

	 auto & p1 = tmp._ptr;
	 uchar * data1 = reinterpret_cast<uchar *>(p1.getCPU());

	 for (auto z = origin.z; z < end.z; z++) {
		 for (auto y = origin.y; y < end.y; y++) {
			 auto x = origin.x;
			 auto i0 = linearIndex(_dim, x, y, z);
			 auto i1 = linearIndex(newDim, x - origin.x, y - origin.y, z - origin.z);
			 memcpy(data1 + stride * i1, data0 + stride * i0, stride * newDim.x);
		 }
	 }

	 p1.commit();*/

	 
	 *this = std::move(getSubvolume(origin, newDim));

}



FAST_EXPORT Volume fast::Volume::getSubvolume(ivec3 origin, ivec3 newDim) const
{
	const ivec3 end = origin + newDim;
	assert(end.x <= _dim.x);
	assert(end.y <= _dim.y);
	assert(end.z <= _dim.z);


	Volume tmp = Volume(newDim, _type);

	const auto & p0 = _ptr;
	const auto stride = p0.stride();
	const uchar * data0 = reinterpret_cast<const uchar *>(p0.getCPU());

	auto & p1 = tmp._ptr;
	uchar * data1 = reinterpret_cast<uchar *>(p1.getCPU());

	//Copy slice by slice
	for (auto z = origin.z; z < end.z; z++) {
		for (auto y = origin.y; y < end.y; y++) {
			auto x = origin.x;
			auto i0 = linearIndex(_dim, x, y, z);
			auto i1 = linearIndex(newDim, x - origin.x, y - origin.y, z - origin.z);
			memcpy(data1 + stride * i1, data0 + stride * i0, stride * newDim.x);
		}
	}

	p1.commit();

	return tmp;
}

FAST_EXPORT void fast::Volume::getSlice(Dir dir, int index, void * output) const
{

	const size_t elemBytes = primitiveSizeof(_type);
	const ivec2 sliceDim = getSliceDim(dir);
	const size_t sliceByteSize = sliceDim.x * sliceDim.y * elemBytes;

	const uchar * data = reinterpret_cast <const uchar*>(getPtr().getCPU());

	if (dir == Z_NEG || dir == Z_POS) {
		if (dir == Z_NEG)
			index = dim().z - index;

		auto ptr = data + sliceByteSize * index;		
		memcpy(output, ptr, sliceByteSize);
	}
	else {
		
		int dirPrimary = getDirIndex(dir);
		int dirSecondary[2] = { (dirPrimary + 1) % 3, (dirPrimary + 2) % 3 };
		auto d = dim();

		ivec2 strideSlice = { 1, d[dirSecondary[0]] };
		
		for (int x = 0; x < d[dirSecondary[0]]; x++) {
			for (int y = 0; y < d[dirSecondary[1]]; y++) {
				ivec3 pos;
				pos[dirPrimary] = index;
				pos[dirSecondary[0]] = x;
				pos[dirSecondary[1]] = y;

				size_t sliceIndex = x * strideSlice.x + y * strideSlice.y;
				size_t volumeIndex = linearIndex(d, pos);

				memcpy(
					((uchar *)output) + sliceIndex * elemBytes,
					data + volumeIndex * elemBytes,
					elemBytes
				);

			}
		}
		
	}


}

FAST_EXPORT ivec2 fast::Volume::getSliceDim(Dir dir) const
{
	int k = getDirIndex(dir);
	ivec3 d = dim();
	return { d[(k + 1) % 3], d[(k + 2) % 3] };
}

void fast::Volume::sum(void * result)
{
	assert(result != nullptr);

	const size_t reduceN = Volume_Reduce_RequiredBufferSize(dim().x * dim().y * dim().z);

	PrimitiveType reductionType = _type;
	if (_type == TYPE_CHAR || _type == TYPE_UCHAR || _type == TYPE_INT || _type == TYPE_UINT) {
		reductionType = TYPE_UINT64;
	}
	
	DataPtr aux;
	aux.alloc(reduceN, primitiveSizeof(reductionType));	

	Volume_Reduce(*getCUDAVolume(), REDUCE_OP_SUM, reductionType, aux.gpu, aux.cpu, result);

}

FAST_EXPORT void fast::Volume::min(void * result)
{
	assert(result != nullptr);

	const size_t reduceN = Volume_Reduce_RequiredBufferSize(dim().x * dim().y * dim().z);

	PrimitiveType reductionType = _type;	
	DataPtr aux;
	aux.alloc(reduceN, primitiveSizeof(reductionType));

	Volume_Reduce(*getCUDAVolume(), REDUCE_OP_MIN, reductionType, aux.gpu, aux.cpu, result);

}

FAST_EXPORT void fast::Volume::max(void * result)
{
	assert(result != nullptr);

	const size_t reduceN = Volume_Reduce_RequiredBufferSize(dim().x * dim().y * dim().z);

	PrimitiveType reductionType = _type;
	DataPtr aux;
	aux.alloc(reduceN, primitiveSizeof(reductionType));

	Volume_Reduce(*getCUDAVolume(), REDUCE_OP_MAX, reductionType, aux.gpu, aux.cpu, result);
}

size_t fast::Volume::sumZeroElems() const
{
	const size_t reduceN = Volume_Reduce_RequiredBufferSize(dim().x * dim().y * dim().z);
	DataPtr aux;
	aux.alloc(reduceN, primitiveSizeof(TYPE_UINT64));	

	uint64 result;
	Volume_Reduce(*getCUDAVolume(), REDUCE_OP_SUM_ZEROELEM, TYPE_UINT64, aux.gpu, aux.cpu, &result);

	return size_t(result);

}

void fast::Volume::clear()
{
	_ptr.clear();	
}



const fast::Texture3DPtr & fast::Volume::getPtr() const
{
	return _ptr;
}



uint fast::Volume::dimInDirection(Dir dir)
{
	return dim()[getDirIndex(dir)];
}

uint fast::Volume::sliceElemCount(Dir dir)
{
	uint index = getDirIndex(dir);
	return dim()[(index + 1) % 3] * dim()[(index + 2) % 3];
}

template <typename T>
void _sumInDir(ivec3 dim, Dir dir, T * in, T * out){
	
	int primary = getDirIndex(dir);
	int secondary[2] = { (primary + 1) % 3, (primary + 2) % 3 };
	int sgn = getDirSgn(dir);

//	#pragma omp parallel
	for (auto i = 0; i < dim[primary]; i++) {

		T sum = 0.0f;
		for (auto j = 0; j < dim[secondary[0]]; j++) {
			for (auto k = 0; k < dim[secondary[1]]; k++) {

				ivec3 pos;
				pos[primary] = i;
				pos[secondary[0]] = j;
				pos[secondary[1]] = k;

				sum += in[linearIndex(dim, pos)];
			}
		}

		if(sgn > 0)
			out[i] = sum;	
		else
			out[dim[primary] - 1 - i] = sum;
	}

}

void fast::Volume::sumInDir(Dir dir, void * output)
{
	auto & ptr = getPtr();
	ptr.allocCPU();
	ptr.retrieve();
		
	
	if (_type == TYPE_FLOAT) {
		_sumInDir<float>(_dim, dir, static_cast<float*>(ptr.getCPU()), static_cast<float*>(output));
	}
	else if (_type == TYPE_DOUBLE) {
		_sumInDir<double>(_dim, dir, static_cast<double*>(ptr.getCPU()), static_cast<double*>(output));
	}	
	else {
		assert("NOT IMPLEMENTED");
	}

}

ivec3 fast::Volume::dim() const
{
	return _dim;
}

PrimitiveType fast::Volume::type() const
{
	return _type;
}

size_t fast::Volume::totalElems() const
{
	return dim().x * dim().y * dim().z;
}


FAST_EXPORT Volume fast::Volume::op(const Volume & other, VolumeOp op)
{
	assert(other.dim().x == dim().x && other.dim().y == dim().y && other.dim().z == dim().z);

	//Todo on gpu
	Volume vol(dim(), _type);
	getPtr().retrieve();
	const uchar * cpu0 = (const uchar*)getPtr().getCPU();
	const uchar * cpu1 = (const uchar*)other.getPtr().getCPU();
	uchar * res = (uchar*)vol.getPtr().getCPU();

	
	auto d = dim();
	const size_t N = d.x * d.y * d.z;

	if (op == VO_MIN) {
#pragma parallel for
		for (auto i = 0; i < N; i++) {
			if (cpu0[i] && cpu1[i]) {
				res[i] = 255;
			}
			else {
				res[i] = 0;
			}
			//res[i] = std::min(cpu0[i], cpu1[i]);
		}
	}
	else if (op == VO_MAX) {
#pragma parallel for
		for (auto i = 0; i < N; i++) {
			res[i] = std::max(cpu0[i], cpu1[i]);
		}
	}
	else if (op == VO_SUB) {
#pragma parallel for
		for (auto i = 0; i < N; i++) {			
			if (cpu0[i] > 0 && cpu1[i] > 0)
				res[i] = 0;
			else
				res[i] = cpu0[i];			
		}
	}
	vol.getPtr().commit();
	return vol;
}

template <typename T> 
void pad(ivec3 res0, ivec3 padMin, ivec3 padMax, const void * data0Ptr, void * data1Ptr, T val) {
	const T * data0 = static_cast<const T*>(data0Ptr);
	T * data1 = static_cast<T*>(data1Ptr);

	ivec3 res1 = res0 + padMin + padMax;
	ivec3 rmax = res0 + padMin;
	#pragma omp parallel for
	for (auto z = 0; z < res1.z; z++) {
		for (auto y = 0; y < res1.y; y++) {
			for (auto x = 0; x < res1.x; x++) {
				auto index1 = linearIndex(res1, { x,y,z });
				if (x < padMin.x || y < padMin.y || z < padMin.z ||
					x >= rmax.x || y >= rmax.y || z >= rmax.z) {
					data1[index1] = val;
				}
				else {
					auto index0 = linearIndex(res0, { x - padMin.x,y - padMin.y ,z - padMin.z });
					data1[index1] = data0[index0];
				}
			}
		}
	}


}

fast::Volume fast::Volume::withZeroPadding(ivec3 paddingMin, ivec3 paddingMax)
{
	//Todo on gpu
	ivec3 res = dim() + paddingMin + paddingMax;
	auto vol = Volume(res, _type);

	getPtr().retrieve();
	void * cpu0 = getPtr().getCPU();

	void * cpu1 = vol.getPtr().getCPU();

	if (_type == TYPE_UCHAR) {
		pad<uchar>(dim(), paddingMin, paddingMax, cpu0, cpu1, 0);
	}
	else if (_type == TYPE_FLOAT) {
		pad<float>(dim(), paddingMin, paddingMax, cpu0, cpu1, 0.0f);
	}
	else if (_type == TYPE_DOUBLE) {
		pad<double>(dim(), paddingMin, paddingMax, cpu0, cpu1, 0.0);
	}
	else {
		assert(false);
	}

	vol.getPtr().commit();

	return vol;
}
FAST_EXPORT Volume fast::Volume::clone()
{
	//Todo on gpu
	Volume vol(dim(), _type);
	getPtr().retrieve();
	void * cpu0 = getPtr().getCPU();
	void * cpu1 = vol.getPtr().getCPU();	
	memcpy(cpu1, cpu0, totalElems() * primitiveSizeof(_type));
	vol.getPtr().commit();
	return vol;
}

void fast::Volume::binarize(float threshold)
{
	launchBinarizeKernel(
		make_uint3(dim().x, dim().y, dim().z),
		getPtr().getSurface(),
		type(),
		threshold
	);
}

CUDA_Volume * fast::Volume::getCUDAVolume() const
{

	if (!_cudaVolumeImpl)
		_cudaVolumeImpl = std::make_unique<CUDA_Volume>();

	
	_cudaVolumeImpl->surf = getPtr().getSurface();
	_cudaVolumeImpl->res = make_uint3(dim().x, dim().y, dim().z);
	_cudaVolumeImpl->type = type();
	_cudaVolumeImpl->tex = getPtr().getTexture();

	return _cudaVolumeImpl.get();	
}

bool fast::Volume::enableOpenGLInterop = false;

