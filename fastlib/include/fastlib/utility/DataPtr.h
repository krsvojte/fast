#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/Types.h>

#include <cuda_runtime.h>
#include <vector>

namespace fast {

	
	

	struct DataPtr {
		void * cpu;
		void * gpu;
		size_t stride; //byte stride between elements		
		
		size_t num;

		FAST_EXPORT DataPtr();

		FAST_EXPORT DataPtr(const DataPtr &) = delete;
		FAST_EXPORT DataPtr & operator = (const DataPtr &) = delete;

		FAST_EXPORT DataPtr(DataPtr &&other);
		FAST_EXPORT DataPtr & operator = (DataPtr &&other);

		FAST_EXPORT size_t byteSize() const { return num*stride; }

		FAST_EXPORT bool memsetDevice(int value = 0);
		
		//Commit to device (offset in bytes)
		FAST_EXPORT bool commit(size_t offset, size_t size);
		//Commits all
		FAST_EXPORT bool commit();

		//Retrieve from device (offset in bytes)
		FAST_EXPORT bool retrieve(size_t offset, size_t size);
		//Retrieves all
		FAST_EXPORT bool retrieve();

		//Simple alloc
		FAST_EXPORT bool allocHost();
		FAST_EXPORT bool allocDevice(size_t num, size_t stride);

		//Allocates both host and device memory
		FAST_EXPORT bool alloc(size_t num, size_t stride);

		FAST_EXPORT ~DataPtr();

	private:
		void _free();

	};

	
	struct Texture3DPtr {	
		FAST_EXPORT Texture3DPtr();
		FAST_EXPORT ~Texture3DPtr();

		FAST_EXPORT Texture3DPtr(const Texture3DPtr &) = delete;
		FAST_EXPORT Texture3DPtr & operator = (const Texture3DPtr &) = delete;

		FAST_EXPORT Texture3DPtr(Texture3DPtr &&other);
		FAST_EXPORT Texture3DPtr & operator = (Texture3DPtr &&other);
				
		/*
			Returns host memory
		*/
		FAST_EXPORT void * getCPU() { return _cpu.ptr; }
		FAST_EXPORT const void * getCPU() const { return _cpu.ptr; }

		/*
			Returns GPU cudaArray
		*/
		FAST_EXPORT const cudaArray * getGPUArray() const { return _gpu; }
		FAST_EXPORT cudaArray * getGPUArray() { return _gpu; }
		FAST_EXPORT bool mapGPUArray();
		FAST_EXPORT bool unmapGPUArray();

		/*
			OpenGL ID of texture
		*/
		FAST_EXPORT uint getGlID() const { return _glID; }
			
		/*
			Total number of elements
		*/
		FAST_EXPORT uint64 num() const {
			return _extent.width * _extent.height * _extent.depth;
		}

		/*
			Total bytesize
		*/
		FAST_EXPORT uint64 byteSize() const {
			return num() * stride(); 
		}

		/*
			Size of element
		*/
		FAST_EXPORT uint64 stride() const {
			return (_desc.x  + _desc.y + _desc.z + _desc.w) / 8;
		}

		FAST_EXPORT ivec3 dim() const {
			return { _extent.width, _extent.height, _extent.depth };
		}

		/*
			Allocates 3D array 
		*/
		FAST_EXPORT bool alloc(PrimitiveType type, ivec3 dim, bool alsoOnCPU = false);

		/*
			Allocates 3D array using OpenGL interop
		*/
		FAST_EXPORT bool allocOpenGL(PrimitiveType type, ivec3 dim, bool alsoOnCPU = false);

		FAST_EXPORT bool allocCPU();

				
		/*
			Commits host memory to device
		*/
		FAST_EXPORT bool commit();

		/*
			Retrieves device memory to host
		*/
		FAST_EXPORT bool retrieve();



		
		/*
			Returns cuda surface handle
		*/
		FAST_EXPORT cudaSurfaceObject_t getSurface() const {
			return _surface; 
		}

		FAST_EXPORT cudaTextureObject_t getTexture() const {
			return _texture;
		}

		/*
			Copies cuda surface handle to specified device memory
		*/
		FAST_EXPORT bool copySurfaceTo(void * gpuSurfacePtr) const;

		/*
			Copies data to linear global memory on device
		*/
		FAST_EXPORT bool copyTo(DataPtr & ptr);

		FAST_EXPORT bool copyFrom(DataPtr & ptr);

		/*
			Clears both cpu & gpu with val
			TODO: memset on gpu instead of doing cpu->gpu copy (i.e. using kernel/memset3d)
		*/
		FAST_EXPORT bool clear(uchar val = 0);
		FAST_EXPORT bool clearGPU(uchar val = 0);

		//Fills volume with elem of type primitivetype
		FAST_EXPORT bool fillSlow(void * elem);

		FAST_EXPORT PrimitiveType type() const { return _type; }

		FAST_EXPORT bool createTexture();


		FAST_EXPORT bool hasTextureObject() const {
			return _textureCreated;
		}
		
	private:

		void _free();
		/*
			Creates surface object
		*/
		bool createSurface();		

		/*
			Sets channel description depending on type
		*/
		void setDesc(PrimitiveType type);

		cudaPitchedPtr _cpu;
		cudaArray * _gpu;
		cudaGraphicsResource * _gpuRes; //for GL  interop
		cudaSurfaceObject_t _surface;

		bool _textureCreated;
		cudaTextureObject_t _texture;

		cudaChannelFormatDesc _desc;
		cudaExtent _extent;
		uint _glID;
		PrimitiveType _type;	

		bool _usesOpenGL;
		
	};

	struct CUDA_VBO {

		FAST_EXPORT CUDA_VBO(uint vbo);
		FAST_EXPORT ~CUDA_VBO();

		FAST_EXPORT CUDA_VBO(const CUDA_VBO &) = delete;
		FAST_EXPORT CUDA_VBO & operator = (const CUDA_VBO &) = delete;

		FAST_EXPORT CUDA_VBO(CUDA_VBO &&other);
		FAST_EXPORT CUDA_VBO & operator = (CUDA_VBO &&other);

		FAST_EXPORT void * getPtr() {
			return _ptr;
		}

		FAST_EXPORT const void * getPtr() const {
			return _ptr;
		}

		FAST_EXPORT uint getVBO() const {
			return _vbo;
		}

		FAST_EXPORT void retrieveTo(void * ptr) const;

		

		struct DefaultAttrib {
			float pos[3]; 
			float normal[3];
			float uv[2];
			float color[4];
		};

		FAST_EXPORT static bool saveObj(const std::vector<DefaultAttrib> & data,  const char * filename);
		FAST_EXPORT bool saveObj(const char * filename) const;


	private:

		void _free();
		uint _vbo;
		cudaGraphicsResource_t _resource;
		void * _ptr;
		size_t _bytes;		
	};

	/*
		Allocates an OpenGL VBO and maps it for CUDA.
		! The structure does not own the vbo, it must be destroyed manually. !
	*/
	FAST_EXPORT CUDA_VBO createMappedVBO(size_t bytesize);

	

}