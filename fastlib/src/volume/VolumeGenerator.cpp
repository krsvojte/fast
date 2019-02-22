#include "volume/VolumeGenerator.h"
#include "volume/VolumeMeasures.h"

#include "utility/RandomGenerator.h"

#include <cstring>
#include <iostream>

#include <glm/gtx/norm.hpp>

namespace fast {


	RNGUniformFloat uniformDist(0, 1);

	float pointToNormBoxDistance(const vec3 & pt){		
		float x = glm::min(pt.x, 1.0f - pt.x);
		float y = glm::min(pt.y, 1.0f - pt.y);
		float z = glm::min(pt.z, 1.0f - pt.z);
		return glm::min(x, glm::min(y, z));
	}

	template<bool FLOOR = true>
	ivec3 normToDiscrete(const ivec3 & dim, const vec3 & normpos) {
		if (FLOOR) {
			return {
				static_cast<int>((dim.x - 1) * normpos.x),
				static_cast<int>((dim.y - 1) * normpos.y),
				static_cast<int>((dim.z - 1) * normpos.z),
			};
		}
		else {
			return {
				static_cast<int>(glm::ceil((dim.x - 1) * normpos.x)),
				static_cast<int>(glm::ceil((dim.y - 1) * normpos.y)),
				static_cast<int>(glm::ceil((dim.z - 1) * normpos.z)),
			};
		}
	}

	vec3 discreteToNorm(const ivec3 & dim, const ivec3 & discretePos) {
		return {
			static_cast<float>(discretePos.x) / dim.x,
			static_cast<float>(discretePos.y) / dim.y,
			static_cast<float>(discretePos.z) / dim.z,
		};
	}



	FAST_EXPORT std::vector<Sphere> generateSpheres(const GeneratorSphereParams & p)
	{
		
		
		if (p.withinBounds) {
			assert(p.rmin <= 0.5f);
		}
				
		int tries = 0;		

		float rrange = p.rmax - p.rmin;
		if (rrange < 0.0f) rrange = -rrange;

		
		vec3 posMax = { 1,1,1 };
		vec3 posMin = { 0,0,0 };
		
		if (p.withinBounds) {
			posMin += p.rmin;
			posMax -= p.rmin;
		}

		vec3 posRange = posMax - posMin;

		
		
		auto collision = [](const Sphere & a, const Sphere & b) -> bool{
			return glm::length2(a.pos - b.pos) < (a.r + b.r) * (a.r + b.r);

		};

		std::vector<Sphere> spheres;

	

		while (spheres.size() < p.N && (p.maxTries == -1 || tries < p.maxTries)) {

			Sphere s;
			s.pos = vec3(uniformDist.next() * posRange.x, uniformDist.next() * posRange.y, uniformDist.next() * posRange.z) + posMin;			

			if (p.withinBounds) {
				float thisMaxR = glm::min(pointToNormBoxDistance(s.pos), p.rmax);			
				s.r = (uniformDist.next() * (thisMaxR - p.rmin)) + p.rmin;
			}
			else {
				s.r = (uniformDist.next() * rrange) + p.rmin;
			}		

			if(!p.overlapping){				
				bool isOk = true;
				for (auto & other : spheres) {					
					if (collision(s, other)) {
						isOk = false;
						break;
					}
				}

				if (!isOk) {
					tries++;
					continue;
				}

			}

			

			s.r2 = s.r*s.r;
			spheres.push_back(s);
		}

		if (p.maxTries > 0 && tries >= p.maxTries) {
			
			return std::vector<Sphere>();
		}

		return spheres;

		

		

	

		
		

	}

	
	FAST_EXPORT float findMaxRandomSpheresRadius(const GeneratorSphereParams & origParams, float tolerance, std::vector<Sphere> * outSpheres)
	{

		std::vector<Sphere> spheres;

		GeneratorSphereParams p = origParams;

		float r0 = 0.0f;		
		float r1 = 0.5f;

		

		while (true) {
			float r = (r0 + r1) / 2.0f;
			p.rmax = r;
			p.rmin = r;		
			
			std::cout << r << ", ";
			spheres = generateSpheres(p);
			
			if (spheres.size() == 0) {
				r1 = r;
			}
			else {
				if (outSpheres)
					*outSpheres = std::move(spheres);

				r0 = r;
			}

			if (glm::abs(r0 - r1) < tolerance) {
				return r0;
			}
		}

		std::cout << std::endl;
	}

	FAST_EXPORT double spheresAnalyticTortuosity(const GeneratorSphereParams & p, const std::vector<Sphere> & spheres)
	{
		
		const double alpha = 0.5;
		double totalArea = 0.0;
		double totalVolume = 0.0f;
		for (auto & s : spheres) {
			totalArea += 4.0f * glm::pi<double>() * s.r2;
			totalVolume += (4.0f / 3.0f) * glm::pi<double>() * s.r2 * s.r;
		}

		if (!p.withinBounds) {
			assert("Not implemented");
			return 0;
		}
			
		return glm::pow(1.0 - totalVolume, -alpha);

		

		/*if (p.withinBounds) {
			std::cout << "[ANALYTIC] Total surface area: " << totalArea << ", volume: " << totalVolume << " porosity: " << 1.0 - totalVolume << std::endl;
			std::cout << "[ANALYTIC] Shape Factor: " << getShapeFactor(totalArea / spheres.size(), totalVolume / spheres.size()) << std::endl;
			std::cout << "[ANALYTIC] Tortuosity: " << glm::pow(1.0-totalVolume, -alpha) << std::endl;
		}*/

		/*double porosity = getPorosity<double>(c);
		std::cout << "[ANALYTIC/DISCRETE EPS] Tortuosity: " << glm::pow(porosity, -alpha) << std::endl;*/

		
	}

	FAST_EXPORT Volume rasterizeSpheres(ivec3 res, const std::vector<Sphere> & spheres)
	{

		Volume c(res, TYPE_UCHAR);

		auto & ptr = c.getPtr();
		ptr.allocCPU();

		const uchar OCCUPIED = 255;
		const uchar EMPTY = 0;
		uchar * data = static_cast<uchar *>(ptr.getCPU());
		memset(data, EMPTY, res.x*res.y*res.z * sizeof(uchar));


		//Rasterize
		for (auto & s : spheres) {
			ivec3 boundMin = normToDiscrete<true>(res, s.pos - vec3(s.r));
			ivec3 boundMax = normToDiscrete<true>(res, s.pos + vec3(s.r));
			
			boundMin = glm::clamp(boundMin, ivec3(0), res - ivec3(1));
			boundMax = glm::clamp(boundMax, ivec3(0), res - ivec3(1));

			for (auto z = boundMin.z; z <= boundMax.z; z++) {
				for (auto y = boundMin.y; y <= boundMax.y; y++) {
					for (auto x = boundMin.x; x <= boundMax.x; x++) {
						vec3 pos = discreteToNorm(res, { x,y,z });

						if (glm::length2(pos - s.pos) < s.r2) {
							data[linearIndex(res, { x,y,z })] = OCCUPIED;
						}

					}
				}
			}


		}

		ptr.commit();

		return c;

	}

	FAST_EXPORT Volume generateFilledVolume(ivec3 res, uchar value)
	{
		Volume c(res, TYPE_UCHAR);
		c.getPtr().clear(value);

		return c;
	}

	FAST_EXPORT bool saveSpheres(const std::vector<Sphere> & spheres, std::ostream & f)
	{
		if (!f.good()) return false;
		f << int(spheres.size()) << "\n";
		for (auto & s : spheres) 
			f << s.pos.x << " " << s.pos.y << " " << s.pos.z << " " << s.r << std::endl;

		return true;

	}

	FAST_EXPORT std::vector<fast::Sphere> loadSpheres(std::istream & f)
	{
		std::vector<fast::Sphere> spheres;
		int N;
		f >> N;
		
		spheres.resize(N);
		for (auto i = 0; i < N; i++) {
			f >> spheres[i].pos.x >> spheres[i].pos.y >> spheres[i].pos.z >> spheres[i].r;
			spheres[i].r2 = spheres[i].r*spheres[i].r;
		}
		return spheres;

	}

}