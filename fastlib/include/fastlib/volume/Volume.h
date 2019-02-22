#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/DataPtr.h>
#include <fastlib/utility/Types.h>

#include <array>
#include <vector>
#include <memory>

struct CUDA_Volume;

namespace fast{

	


	class Volume {
	public:

		FAST_EXPORT Volume(
			ivec3 dim, 
			PrimitiveType type 						
			);
		
		FAST_EXPORT Volume(
			Texture3DPtr && ptr			
			);

// 		FAST_EXPORT Volume(
// 			Volume & v
// 		);

		FAST_EXPORT Volume & operator=(const Volume & other) = delete;
		FAST_EXPORT Volume(const Volume & other) = delete;
		FAST_EXPORT Volume(Volume && other); //todo..
		FAST_EXPORT Volume & operator= (Volume && other);
		FAST_EXPORT ~Volume();

		FAST_EXPORT Texture3DPtr & getPtr();
		
		FAST_EXPORT const Texture3DPtr & getPtr() const;

		FAST_EXPORT Volume clone();
		

		

		FAST_EXPORT void resize(ivec3 origin, ivec3 dim);

		//On CPU, assumes ptr data was retrieved
		FAST_EXPORT Volume getSubvolume(ivec3 origin, ivec3 dim) const;

		FAST_EXPORT void getSlice(Dir dir, int index, void * output) const;
		FAST_EXPORT ivec2 getSliceDim(Dir dir) const;
		

		FAST_EXPORT void sum(void * result);
		FAST_EXPORT void min(void * result);
		FAST_EXPORT void max(void * result);

		
		FAST_EXPORT size_t sumZeroElems() const;

		/*
			Clears both buffers
		*/
		FAST_EXPORT void clear();
	
		
		FAST_EXPORT uint dimInDirection(Dir dir);
		FAST_EXPORT uint sliceElemCount(Dir dir);

		FAST_EXPORT void sumInDir(Dir dir, void * output);

		
		FAST_EXPORT ivec3 dim() const;
		FAST_EXPORT PrimitiveType type() const;
		FAST_EXPORT size_t totalElems() const;

		enum VolumeOp{
			VO_MIN,VO_MAX,VO_SUB
		};
		FAST_EXPORT Volume op(const Volume & vol, VolumeOp op);
		
		FAST_EXPORT Volume withZeroPadding(ivec3 paddingMin, ivec3 paddingMax);
		
		FAST_EXPORT void binarize(float threshold = 1.0f);

		FAST_EXPORT CUDA_Volume * getCUDAVolume() const;

		FAST_EXPORT static bool enableOpenGLInterop;

	private:		
		ivec3 _dim;
		PrimitiveType _type;
		//bool _doubleBuffered;
		//uchar _current;

		Texture3DPtr _ptr;
		mutable std::unique_ptr<CUDA_Volume> _cudaVolumeImpl; //todo not mutable
		
		//std::string _name;
		
	};

	

}
