#include "utility/DataPtr.h"

#include "cuda/CudaUtility.h"
#include "utility/GLGlobal.h"


#include <glm/gtx/hash.hpp>
#include <unordered_map>
#include <fstream>

#include <assert.h>
#include <cuda_gl_interop.h>

#include <cstring>




//#define GPU_MEMORY_TRACE

#ifdef GPU_MEMORY_TRACE
#include <iostream>
#endif


#ifdef TRACK_GPU_ALLOC
#include <iostream>
#endif


fast::DataPtr::DataPtr()
{
	memset(this, 0, sizeof(DataPtr));	
}

fast::DataPtr::~DataPtr()
{
	_free();
}

void fast::DataPtr::_free()
{
	if (cpu) {
		delete[]((uchar*)cpu);
	}
	if (gpu) {
		_CUDA(cudaFree(gpu));
	}

	memset(this, 0, sizeof(DataPtr));
}

fast::DataPtr & fast::DataPtr::operator=(fast::DataPtr &&other)
{
	if (this != &other) {
		this->_free();
		memcpy(this, &other, sizeof(other));
		memset(&other, 0, sizeof(DataPtr));
	}
	return *this;
}

fast::DataPtr::DataPtr(fast::DataPtr &&other)
{
	memcpy(this, &other, sizeof(other));
	memset(&other, 0, sizeof(DataPtr));
}

bool fast::DataPtr::retrieve(size_t offset, size_t size)
{
	
	assert(gpu != nullptr);
	assert(stride > 0);
	assert(size > 0);

	if (!cpu) {
		allocHost();
	}

	return _CUDA(
		cudaMemcpy((uchar*)cpu + stride * offset, (uchar*)gpu + stride * offset, size, cudaMemcpyDeviceToHost)
	);
}

bool fast::DataPtr::retrieve()
{
	return retrieve(0, byteSize());
}

bool fast::DataPtr::allocHost()
{
	assert(gpu != nullptr);

	if (cpu) {
		delete[] ((uchar*)cpu);
		cpu = nullptr;
	}
	

	cpu = new uchar[num*stride];
	return true;

}

bool fast::DataPtr::allocDevice(size_t num, size_t stride)
{	
#ifdef GPU_MEMORY_TRACE
	{
		size_t bytes = num*stride;
		float MB = bytes / (1024.0f*1024.0f);		
		cudaPrintMemInfo();
		std::cout << "|| DataPtr::allocDevice " << MB << "MB" << std::endl;
		
	}
#endif

	if (gpu) {
		if (num == this->num && stride == this->stride) 
			return true;

		
		_CUDA(cudaFree(gpu));
		gpu = nullptr;
	}

	

	if (_CUDA(cudaMalloc((void **)&gpu, num*stride)) && gpu != nullptr){
		this->stride = stride;
		this->num = num;
		
		//Realloc cpu if existed
		if (cpu) {
			allocHost();
		}

		return true;
	}
	else {
		assert(false);		
	}

	return false;
}


bool fast::DataPtr::alloc(size_t num, size_t stride)
{
	return allocDevice(num, stride) && allocHost();
}



bool fast::DataPtr::memsetDevice(int value)
{	
	return _CUDA(cudaMemset(gpu, value, byteSize()));
}

bool fast::DataPtr::commit(size_t offset, size_t size) {

	assert(cpu != nullptr);
	assert(gpu != nullptr);
	assert(stride > 0);
	assert(size > 0);

	return _CUDA(
		cudaMemcpy((char*)gpu + stride * offset, (char*)cpu + stride * offset, size, cudaMemcpyHostToDevice)
	);
}

bool fast::DataPtr::commit()
{
	return commit(0, byteSize());
}

//////////////////


fast::Texture3DPtr::Texture3DPtr()
{
	memset(this, 0, sizeof(Texture3DPtr));
	_desc.f = cudaChannelFormatKindUnsigned;	
}



fast::Texture3DPtr & fast::Texture3DPtr::operator=(fast::Texture3DPtr &&other)
{
	if (this != &other) {
		this->_free();
		memcpy(this, &other, sizeof(other));
		memset(&other, 0, sizeof(Texture3DPtr));
	}
	return *this;
}

fast::Texture3DPtr::Texture3DPtr(fast::Texture3DPtr &&other)
{

	memcpy(this, &other, sizeof(other));
	memset(&other, 0, sizeof(Texture3DPtr));
}

fast::Texture3DPtr::~Texture3DPtr()
{
	_free();
}

bool fast::Texture3DPtr::alloc(PrimitiveType type, ivec3 dim, bool alsoOnCPU)
{
	
	#ifdef GPU_MEMORY_TRACE
	{
		size_t bytes = dim.x*dim.y*dim.z*primitiveSizeof(type);
		float MB = bytes / (1024.0f*1024.0f);		
		cudaPrintMemInfo();
		std::cout << "|| Texture3DPtr::alloc " << MB << "MB" << std::endl;

	}
	#endif

	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);

	_extent = make_cudaExtent(dim.x, dim.y, dim.z);

	setDesc(type);

	if (!_CUDA(cudaMalloc3DArray(&_gpu, &_desc, _extent, 0)))
		return false;

	_usesOpenGL = false;

	
	createSurface();

	if (alsoOnCPU) {
		return allocCPU();
	}
		
	return true;

}

bool fast::Texture3DPtr::allocOpenGL(PrimitiveType type, ivec3 dim, bool alsoOnCPU /*= false*/)
{
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);
	
#ifdef GPU_MEMORY_TRACE
	{
		size_t bytes = dim.x*dim.y*dim.z*primitiveSizeof(type);
		float MB = bytes / (1024.0f*1024.0f);		
		cudaPrintMemInfo();
		std::cout << "|| Texture3DPtr::allocOpenGL " << MB << "MB" << std::endl;
	}
#endif
	

	if (_gpu != nullptr && _type == type) {
		auto newExtent = make_cudaExtent(dim.x, dim.y, dim.z);
		if (_extent.depth == newExtent.depth && _extent.height == newExtent.height && _extent.width == newExtent.width)
			return true;		
	}

	//
	//assert(_glID == 0); //no realloc yet
	//assert(_gpu == nullptr);
	//


	_extent = make_cudaExtent(dim.x, dim.y, dim.z);
	
	//Set channel descriptor (elements, sizes)
	
	setDesc(type);
	
	
	if (alsoOnCPU) {		
		allocCPU();
	}		
		
	GL(glGenTextures(1, (GLuint*)&_glID));
	GL(glBindTexture(GL_TEXTURE_3D, _glID));
	
	GL(glPixelStorei(GL_PACK_ALIGNMENT, 4));
	GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));


	const bool linear = true;
	if (linear) {
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	}
	else {
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	}
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
	
	switch (type) {
		case TYPE_UCHAR:
		case TYPE_CHAR:	
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, dim.x, dim.y, dim.z, 0, GL_RED, GL_UNSIGNED_BYTE, NULL));	
			break;	
		case TYPE_UCHAR4:
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, dim.x, dim.y, dim.z, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
			break;
		case TYPE_FLOAT:	
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, dim.x, dim.y, dim.z, 0, GL_RED, GL_FLOAT, NULL));
			break;
		case TYPE_DOUBLE:
		case TYPE_UINT64:
			//Double or uint64 ~ reinterpreted int2
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32I, dim.x, dim.y, dim.z, 0, GL_RG_INTEGER, GL_UNSIGNED_INT, NULL));
			break;
		case TYPE_INT:					
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32I, dim.x, dim.y, dim.z, 0, GL_RED_INTEGER, GL_INT, NULL));
			break;
		case TYPE_UINT:			
			GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, dim.x, dim.y, dim.z, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL));			
			break;
		default:
			assert("Not implemented");
			break;
	};
	

	//Register texture for interop
	GL(_CUDA(cudaGraphicsGLRegisterImage(&_gpuRes, _glID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore)));

	//Map resource for cuda
	_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));

	//Get pointer
	_CUDA(cudaGraphicsSubResourceGetMappedArray(&_gpu, _gpuRes, 0, 0));
	
	//Create surface
	createSurface();

	//Unmap resource
	_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));	

	_usesOpenGL = true;
	
	
	return true;
}

bool fast::Texture3DPtr::allocCPU(){
	assert(byteSize() > 0);

	//Already allocated
	if (_cpu.ptr)
		return false;

	void * hostPtr = new char[byteSize()];
	_cpu = make_cudaPitchedPtr(hostPtr, stride() * _extent.width, _extent.width, _extent.height);
	return true;
}

bool fast::Texture3DPtr::mapGPUArray()
{
	//Map resource for cuda
	_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));

	//Get pointer
	return _CUDA(cudaGraphicsSubResourceGetMappedArray(&_gpu, _gpuRes, 0, 0));
}

 bool fast::Texture3DPtr::unmapGPUArray()
{
	 //Unmap resource
	 return _CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));
}

bool fast::Texture3DPtr::commit()
{

	assert(_cpu.ptr != nullptr);
	assert(_gpu != nullptr);

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;
	p.kind = cudaMemcpyHostToDevice;
	
	p.srcPtr = _cpu;	
	p.dstArray = _gpu;
	
	if(_glID != 0)
		_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));

	bool res = _CUDA(cudaMemcpy3D(&p));
	
	if (_glID != 0)
		_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));
	return res;


	/*
	//Alternatively use glTexImage3D
		GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, _extent.width, _extent.height, _extent.depth, 0, GL_RED, GL_FLOAT, _cpu.ptr));
		return true;
	*/
}


bool fast::Texture3DPtr::retrieve()
{

	assert(_cpu.ptr != nullptr);
	assert(_gpu != nullptr);

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;	
	//p.extent.width *= stride();
	
	p.kind = cudaMemcpyDeviceToHost;	

	p.srcArray = _gpu;
	p.dstPtr = _cpu;	

	if (_glID != 0)
		_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));
	bool res = _CUDA(cudaMemcpy3D(&p));

	if (_glID != 0)
		_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));

	return res;
}

bool fast::Texture3DPtr::copySurfaceTo(void * gpuSurfacePtr) const
{
	return _CUDA(
		cudaMemcpy(gpuSurfacePtr, &_surface, sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice)
	);
}

bool fast::Texture3DPtr::copyTo(DataPtr & ptr)
{
	assert(ptr.byteSize() == byteSize());
	assert(ptr.stride == stride());
	bool res = true;
	res &= mapGPUArray();

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;
	p.kind = cudaMemcpyDeviceToDevice;

	p.srcArray = _gpu;

	auto gpuPitched = make_cudaPitchedPtr(
		ptr.gpu, stride() * _extent.width, _extent.width, _extent.height
	);

	p.dstPtr = gpuPitched;

	res &= _CUDA(cudaMemcpy3D(&p));

	res &= unmapGPUArray();
	return res;
}

bool fast::Texture3DPtr::copyFrom(DataPtr & ptr)
{
	assert(ptr.byteSize() == byteSize());
	assert(ptr.stride == stride());
	bool res = true;
	res &= mapGPUArray();

	cudaMemcpy3DParms p;
	memset(&p, 0, sizeof(cudaMemcpy3DParms));

	p.extent = _extent;
	p.kind = cudaMemcpyDeviceToDevice;

	auto gpuPitched = make_cudaPitchedPtr(
		ptr.gpu, stride() * _extent.width, _extent.width, _extent.height
	);

	p.srcPtr = gpuPitched;

	p.dstArray = _gpu;

	res &= _CUDA(cudaMemcpy3D(&p));

	res &= unmapGPUArray();
	return res;
}

bool fast::Texture3DPtr::clear(uchar val /*= 0*/)
{
	memset(getCPU(), val, this->byteSize());	
	return commit();

}

bool fast::Texture3DPtr::clearGPU(uchar val /*= 0*/)
{
	/*_CUDA(cudaGraphicsMapResources(1, &_gpuRes, 0));
	cudaPitchedPtr ptr;
	ptr.ptr = _gpu;
	ptr.pitch = _extent.width * stride();
	ptr.xsize = _extent.width * stride();
	ptr.ysize = _extent.width * _extent.height * stride();
	bool res = _CUDA(cudaMemset3D(ptr, val, _extent));
	_CUDA(cudaGraphicsUnmapResources(1, &_gpuRes, 0));*/
	return false;
	//return res;
}

bool fast::Texture3DPtr::fillSlow(void * elem)
{
	auto * cpu = getCPU();
	
	const auto numElem = num();
	const auto size = primitiveSizeof(_type);
	for (auto i = 0; i < numElem; i++){
		memcpy((char*)cpu + i*size, elem, size);
	}
	
	return commit();

}

void fast::Texture3DPtr::_free()
{
	if (_cpu.ptr) {
		delete[] _cpu.ptr;
	}

	if (_gpu) {
		_CUDA(cudaDestroySurfaceObject(_surface));
		if (_usesOpenGL) {
			_CUDA(cudaGraphicsUnregisterResource(_gpuRes));
			GL(glBindTexture(GL_TEXTURE_3D, _glID));
			GL(glDeleteTextures(1, &_glID));
		}
		else {
			cudaFreeArray(_gpu);
		}
	}

	memset(this, 0, sizeof(Texture3DPtr));
}

bool fast::Texture3DPtr::createSurface()
{
	cudaResourceDesc resDesc; 

	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = _gpu;
	
	return _CUDA(cudaCreateSurfaceObject(&_surface, &resDesc));
}

bool fast::Texture3DPtr::createTexture()
{
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = _gpu;

	cudaTextureDesc td;
	td.normalizedCoords = 1;
	td.addressMode[0] = cudaAddressModeClamp;
	td.addressMode[1] = cudaAddressModeClamp;
	td.addressMode[2] = cudaAddressModeClamp;
	td.readMode = cudaReadModeNormalizedFloat;
	td.sRGB = 0;
	td.filterMode = cudaFilterModeLinear; //cudaFilterModeLinear;
	
	td.maxAnisotropy = 16;
	td.mipmapFilterMode = cudaFilterModeLinear;
	td.minMipmapLevelClamp = 0;
	td.maxMipmapLevelClamp = 0;
	td.mipmapLevelBias = 0;

	//cudaResourceViewDesc resd;

	bool res = _CUDA(cudaCreateTextureObject(&_texture, &resDesc, &td, nullptr));
	if (res)
		_textureCreated = true;
	
	return res;
}

void fast::Texture3DPtr::setDesc(PrimitiveType type)
{
	switch (type) {

	case TYPE_FLOAT3:
		_desc.x = sizeof(float) * 8;
		_desc.y = sizeof(float) * 8;
		_desc.z = sizeof(float) * 8;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_FLOAT4:
		_desc.x = sizeof(float) * 8;
		_desc.y = sizeof(float) * 8;
		_desc.z = sizeof(float) * 8;
		_desc.w = sizeof(float) * 8;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_FLOAT:
		_desc.x = sizeof(float) * 8;
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindFloat;
		break;
	case TYPE_DOUBLE:
	case TYPE_UINT64:
		_desc.x = (sizeof(int2) / 2) * 8;
		_desc.y = (sizeof(int2) / 2) * 8;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindSigned;
		break;
	case TYPE_INT:
		_desc.x = sizeof(int) * 8;
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindSigned;
		break;
	case TYPE_UINT:
		_desc.x = sizeof(uint) * 8;
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindUnsigned;
		break;
	case TYPE_UCHAR4:
		_desc.x = sizeof(uchar) * 8;
		_desc.y = sizeof(uchar) * 8;
		_desc.z = sizeof(uchar) * 8;
		_desc.w = sizeof(uchar) * 8;
		_desc.f = cudaChannelFormatKindUnsigned;
		break;
	case TYPE_UCHAR:
		//
	default:
		_desc.x = sizeof(unsigned char) * 8;
		_desc.y = 0;
		_desc.z = 0;
		_desc.w = 0;
		_desc.f = cudaChannelFormatKindUnsigned;
		break;
	};

	_type = type;

}

fast::CUDA_VBO::CUDA_VBO(uint vbo)
{
	memset(this, 0, sizeof(CUDA_VBO));
	_vbo = vbo;
	_CUDA(cudaGraphicsGLRegisterBuffer(&_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));
	_CUDA(cudaGraphicsMapResources(1, &_resource, 0));
	_CUDA(cudaGraphicsResourceGetMappedPointer(&_ptr, &_bytes, _resource));
	
}


void fast::CUDA_VBO::retrieveTo(void * ptr) const
{
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, _bytes, ptr);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

FAST_EXPORT bool fast::CUDA_VBO::saveObj(const std::vector<DefaultAttrib> & data, const char * filename)
{
	std::vector<ivec3> indices;
	std::vector<vec3> vertices;
	std::vector<vec3> normals;
	std::unordered_map<vec3, size_t, std::hash<glm::vec3>> posToIndices;
	

	for (auto i = 0; i < data.size(); i += 3) {

		ivec3 triangleIndices;

		for (auto k = 0; k < 3; k++) {
			auto & d = data[i + k];
			vec3 v = { d.pos[0],d.pos[1],d.pos[2] };
			vec3 n = { d.normal[0],d.normal[1],d.normal[2] };

			auto it = posToIndices.find(v);
			size_t newIndex = 0;

			if (it == posToIndices.end()) {
				vertices.push_back(v);
				normals.push_back(n);
				newIndex = vertices.size() - 1;
				posToIndices[v] = newIndex;
			}
			else {
				newIndex = it->second;
				normals[newIndex] += n;
			}

			triangleIndices[k] = int(newIndex);
		}
		indices.push_back(triangleIndices);
	}

	//Normalize normals sum
	for (auto & n : normals) {
		n = glm::normalize(n);
	}

	std::ofstream f(filename);
	if (!f.good())
		return false;

	for (auto v : vertices) {
		f << "v " << v.x << " " << v.y << " " << v.z << "\n";
	}
	for (auto n : normals) {
		f << "vn " << n.x << " " << n.y << " " << n.z << "\n";
	}
	for (auto i : indices) {
		f << "f " << i.x + 1 << " " << i.y + 1 << " " << i.z + 1 << "\n";
	}

	return true;
}

FAST_EXPORT bool fast::CUDA_VBO::saveObj(const char * filename) const
{
	std::vector<DefaultAttrib> data(_bytes / sizeof(DefaultAttrib));
	retrieveTo(data.data());	
	return CUDA_VBO::saveObj(data, filename);
}

void fast::CUDA_VBO::_free()
{
	if (_ptr) {
		_CUDA(cudaGraphicsUnmapResources(1, &_resource, 0));
		_CUDA(cudaGraphicsUnregisterResource(_resource));
	}
	memset(this, 0, sizeof(CUDA_VBO));
}

fast::CUDA_VBO & fast::CUDA_VBO::operator=(CUDA_VBO &&other)
{
	if (this != &other) {
		this->_free();
		memcpy(this, &other, sizeof(other));
		memset(&other, 0, sizeof(CUDA_VBO));
	}
	return *this;
}

 fast::CUDA_VBO::CUDA_VBO(CUDA_VBO &&other)
{
	 memcpy(this, &other, sizeof(other));
	 memset(&other, 0, sizeof(CUDA_VBO));
}

fast::CUDA_VBO::~CUDA_VBO()
{
	_free();
}

fast::CUDA_VBO fast::createMappedVBO(size_t bytesize)
{

	
	GLuint vbo;

	
	glGenBuffers(1, &vbo);
#ifdef TRACK_GPU_ALLOC
	std::cout << "GenBuffer " << vbo << ", " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
	glBindBuffer(GL_ARRAY_BUFFER, vbo);	
	glBufferData(GL_ARRAY_BUFFER, bytesize, 0, GL_DYNAMIC_DRAW);
#ifdef TRACK_GPU_ALLOC
	std::cout << "glBufferData " << vbo << ", size: " << bytesize << ", " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
	glBindBuffer(GL_ARRAY_BUFFER, 0);



	return CUDA_VBO(vbo);
}
