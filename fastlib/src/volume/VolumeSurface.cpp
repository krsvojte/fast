#include "volume/VolumeSurface.h"

#include "volume/Volume.h"
#include "cuda/VolumeSurface.cuh"

namespace fast {


	FAST_EXPORT Volume getVolumeArea(const Volume & mask, ivec3 res /*= ivec3(0,0,0)*/, float isovalue /*= 0.5f */)
	{

		if (!mask.getPtr().hasTextureObject()) {
			throw "Mask has to have a texture object. Create it by mask.getPtr().createTextureObject()";
		}

		VolumeSurface_MCParams params;
		if (res.x == 0 || res.y == 0 || res.z == 0) {
			res = mask.dim();
		}
		
		
		params.res = make_uint3(res.x, res.y, res.z);
		params.isovalue = isovalue;
		params.smoothingOffset = 1.0f;

		
		auto maxDim = std::max(res.x, std::max(res.y, res.z));
		params.dx = params.dy = params.dz = 1.0f / maxDim;


		Volume areas(res, TYPE_DOUBLE);
		VolumeSurface_MarchingCubesArea(*mask.getCUDAVolume(), params, *areas.getCUDAVolume());


		return areas;
		
	}

	FAST_EXPORT VolumeAreaMeshOutput getVolumeAreaMesh(const Volume & mask, bool openGLInterop, ivec3 res /*= ivec3(0, 0, 0)*/, float isovalue /*= 0.5f*/)
	{
		if (!mask.getPtr().hasTextureObject()) {
			throw "Mask has to have a texture object. Create it by mask.getPtr().createTextureObject()";
		}

		/*if (NvertsOut == nullptr || vboOut == nullptr) {
			throw "Nullptr output arguments";
		}*/

		VolumeSurface_MCParams params;
		if (res.x == 0 || res.y == 0 || res.z == 0) {
			res = mask.dim();
		}

		params.res = make_uint3(res.x, res.y, res.z);
		params.isovalue = isovalue;
		params.smoothingOffset = 1.0f;

		auto maxDim = std::max(res.x, std::max(res.y, res.z));
		params.dx = params.dy = params.dz = 1.0f / maxDim;

		VolumeAreaMeshOutput out;

		
		VolumeSurface_MarchingCubesMesh(
			*mask.getCUDAVolume(),
			params,
			openGLInterop,
			&out.vbo,
			&out.Nverts,
			out.data
		);

		
		return out;
	}

	
}

