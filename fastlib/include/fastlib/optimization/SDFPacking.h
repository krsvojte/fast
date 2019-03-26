#pragma once

#include <fastlib/geometry/ImplicitGeometry.h>
#include <fastlib/geometry/AABB.h>
#include <fastlib/optimization/SimulatedAnnealing.h>
#include <fastlib/utility/RandomGenerator.h>
#include <distfun/distfun.hpp>

#include <fastlib/geometry/SDF.h>

#include <vector>
#include <memory>


namespace fast {

	class Volume;

	class SDFPacking {		

	public:

		using State = SDFArray;

		struct Params {
			Params() : 
				annealingSchedule(0.999f), 
				maxIter(5000), 
				startingTemperature(1.0f), 
				allowPartialOutside(false), 
				verbose(false)
				{
			}
			float annealingSchedule;
			size_t maxIter;
			float startingTemperature;
			bool allowPartialOutside;
			bool verbose;			

		};

		FAST_EXPORT SDFPacking(
			const AABB & domain = AABB(vec3(-1), vec3(1)),
			bool deterministic = false			
		);

		FAST_EXPORT ~SDFPacking();
		
		enum DegreeOfFreedom : int {
			DOF_POS = 1,
			DOF_ROT = 2,
			DOF_SCALE = 3
		};

		void addPrimitive(const distfun::sdPrimitive & primitive);

		FAST_EXPORT void addEllipse(size_t N, vec3 size);

		FAST_EXPORT void setDegreesOfFreedom(int dof) {
			_dof = dof;
		}

		FAST_EXPORT bool init(const Params & params = Params());
		FAST_EXPORT bool step();

		/*
			Sets volume to rasterize current state to
		*/
		FAST_EXPORT void setRasterVolume(Volume * volptr);
		FAST_EXPORT void rasterize(bool commit = true);
		FAST_EXPORT void rasterizeOverlap(Volume & volume);

		FAST_EXPORT std::vector<AABB> getParticleBounds() const;

		FAST_EXPORT const SimulatedAnnealing<State> & getSA() const {
			return _sa;
		}

		FAST_EXPORT SimulatedAnnealing<State> & getSA() {
			return _sa;
		}
		
		FAST_EXPORT float getMaxPenetrationFraction(int depth = 3, const State * state = nullptr) const;
		FAST_EXPORT float getPorosity(int depth = 4);
	
		FAST_EXPORT void setShowBest(bool val);
	
	private:
		AABB _domain;		
		int _dof;
		Volume * _rasterVolume;

		//std::vector<std::unique_ptr<distfun::Primitive>> _primitives;
		SimulatedAnnealing<State> _sa;
		State _initState;
		
		RNGNormal _normal;
		RNGUniformFloat _uniform;
		RNGUniformInt _uniformInt;

		bool _deterministic;

		Params _currentParams;
		bool _showBest;

		
	};

}