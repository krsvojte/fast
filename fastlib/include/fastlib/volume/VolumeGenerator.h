#pragma once



#include <fastlib/FastLibDef.h>
#include <fastlib/volume/Volume.h>

#include <vector>
#include <iostream>

namespace fast {


	struct GeneratorSphereParams {
		size_t N;
		float rmin;
		float rmax;
		bool overlapping;
		bool withinBounds;
		int maxTries;
	};


	struct Sphere {
		vec3 pos;
		float r;
		float r2;
	};


	FAST_EXPORT std::vector<Sphere> generateSpheres(const GeneratorSphereParams & p);

//	FAST_EXPORT size_t findMaxRandomSpheresN(const GeneratorSphereParams & p);

	FAST_EXPORT float findMaxRandomSpheresRadius(const GeneratorSphereParams & p, float tolerance = 0.001f, std::vector<Sphere> * outSpheres = nullptr);

	FAST_EXPORT double spheresAnalyticTortuosity(const GeneratorSphereParams & p, const std::vector<Sphere> & spheres);

	FAST_EXPORT Volume rasterizeSpheres(ivec3 res, const std::vector<Sphere> & spheres);

	FAST_EXPORT Volume generateFilledVolume(ivec3 res, uchar value);


	
	FAST_EXPORT bool saveSpheres(const std::vector<Sphere> & spheres, std::ostream & f);
	FAST_EXPORT std::vector<Sphere> loadSpheres(std::istream & f);

}