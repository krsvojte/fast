

#include "optimization/SDFPacking.h"
#include "volume/Volume.h"
#include "utility/PoissonSampler.h"

#include "cuda/SDF.cuh"

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include "utility/Timer.h"

#include <iostream>

#define DISTFUN_IMPLEMENTATION
#define DISTFUN_ENABLE_CUDA
#include "distfun/distfun.hpp"




fast::SDFPacking::SDFPacking(const AABB & domain /*= AABB(vec3(-1), vec3(1))*/, bool deterministic)
	:_domain(domain), _dof(DOF_POS), _normal(0.0f,1.0f), _uniform(0.0f,1.0f), _uniformInt(0, INT_MAX),
	_deterministic(deterministic),
	_rasterVolume(nullptr),
	_showBest(false)
{
	if (!_deterministic) {
		_normal.seedByTime();
		_uniform.seedByTime();
		_uniformInt.seedByTime();
	}

}

fast::SDFPacking::~SDFPacking()
{

}

void fast::SDFPacking::addPrimitive(const distfun::Primitive & primitive)
{
	_initState.push_back(primitive);	
}

void fast::SDFPacking::addEllipse(size_t N, vec3 size)
{
	distfun::Primitive p;
	p.invTransform = distfun::mat4(1.0f);
	p.type = distfun::Primitive::SD_ELLIPSOID;
	p.rounding = 0.0f;
	p.params.ellipsoid.size = size;
	for (auto i = 0; i < N; i++) {
		addPrimitive(p);
	}
}







bool fast::SDFPacking::init(const fast::SDFPacking::Params & __params)
{
	_currentParams = __params;
	if (_initState.size() == 0) return false;
	

	//Initial position	
	{
		const size_t N = _initState.size();
		float ncbrt = std::cbrtf(N);
		const auto diag = _domain.diagonal();
		const float r = 2 * glm::min(diag.x, glm::min(diag.y, diag.z)) / (2.0f * ncbrt);
		auto positions = poissonSampler(r, N, _domain);

		for (auto i = 0; i < _initState.size(); i++) {
			vec3 pos = positions[i];
			_initState[i].invTransform = glm::translate(mat4(1.0f), -pos);
		}		
	}
	
	//Test case for 2 elems
	if (_initState.size() == 2) {
		_initState[0].invTransform = glm::translate(mat4(1.0f), -vec3(_initState[0].params.ellipsoid.size.x * 0.001f,0,0) );
		_initState[1].invTransform = glm::translate(mat4(1.0f), vec3(_initState[0].params.ellipsoid.size.x * 0.001f,0,0) );
	}
	

	_sa.score = [&](const State & s) {


		/*
		Show current state (not scored one)
		*/
		static int counter = 0;
		if (_rasterVolume && counter % 1 == 0) {

			if ((_showBest && _sa.bestHasChanged) || !_showBest) {
				rasterize();
			}
		}

		return getMaxPenetrationFraction(3, &s);

		float totalVolume = 0.0f;

		
		for(auto p : s){			
			auto T = glm::inverse(p.invTransform);
			if (p.type == distfun::Primitive::SD_ELLIPSOID) {
				const auto size = p.params.ellipsoid.size;
				const auto scale = vec3(
					glm::length(vec3(T[0])),
					glm::length(vec3(T[1])),
					glm::length(vec3(T[2]))
				);
				const vec3 s = size * scale; 
				totalVolume += s.x * s.y * s.z * (4.0f / 3.0f) * glm::pi<float>();
			}
		}

		/*
			Calculate actual union volume
		*/

		


		///////////////// BBs
		if (_currentParams.allowPartialOutside) {
			float maxDist = glm::length(_domain.diagonal());
			float bbIntersectionVolume = 0.0f;
			for (auto & p : s) {
				auto dbb = primitiveBounds(p, glm::length(_domain.diagonal()) * 0.5f);
				AABB bb = { dbb.min, dbb.max };
				AABB isectbb = bb.getIntersection(_domain);

				if (isectbb.isValid()) {
					bbIntersectionVolume += (bb.volume() - isectbb.volume()) / bb.volume();
				}
				else {
					bbIntersectionVolume += bb.volume() / bb.volume();
				}
			}
			bbIntersectionVolume /= s.size();
		}




		
		
		/*float minPorosity = 1.0f - (totalVolume / _domain.volume());

		float porosity = getPorosity(4);
				
		float overlap = glm::abs(totalVolume - actualVolume) / totalVolume;

		if (_currentParams.verbose) {
			std::cout << "porosity: " << porosity << 
				", min achieavable: " << minPorosity << 
				", best: " << _sa.bestStateScore << 
				", overlap: " << overlap << std::endl;
		}*/
		

		
		//return porosity;
	};

	_sa.getNeighbour = [&](const State & s) {		

		const bool moveRandom = true;
		const bool moveInterElastic = true;
		const bool moveInterElasticCUDA = false;
		const bool moveWallElastic = _currentParams.allowPartialOutside;
		const bool moveRandomAtLowT = true;
		const float lowT = 0.175f;

		std::vector<vec3> moves;
		std::vector<vec3> unattenuatedMoves;
		std::vector<AABB> aabbs;
		moves.resize(s.size(), vec3(0.0f));
		unattenuatedMoves.resize(s.size(), vec3(0.0f));
		aabbs.resize(s.size(), AABB());

		for (auto i = 0; i < s.size(); i++) {
			auto tmp = primitiveBounds(s[i], glm::length(_domain.diagonal()) * 0.5f);;
			aabbs[i] = AABB(tmp.min, tmp.max);
		}

		float transAmount = 0.05f * _sa.currentTemperature();// 0.05f * 0.1f;
		float transProb = 0.05f + 0.15f + 0.5f * _sa.currentTemperature();// + 0.5f * _sa.currentTemperature();
		if(moveRandom){			
			for(auto i = 0; i < s.size(); i++){
				auto & p = s[i];								
				//if (_uniform.next() < transProb) {
					float actualAmount = transAmount * glm::length(aabbs[i].diagonal());
					vec3 move = actualAmount * vec3(_normal.next(), _normal.next(), _normal.next());
					moves[i] += move;
				//}				
			}
		}


		
		//Elasticity
		if(moveInterElastic){

			static float averageMs = 0.0f;
			static int averageN = 0;


			Timer t(true);
			auto res = SDFElasticity(s, { _domain.min, _domain.max }, _uniform, 3);

			averageMs += t.timeMs();
			averageN++;
			//std::cout << "*Elasiticity: " << t.timeMs() << " ms, average: " << averageMs / averageN << " ms" << std::endl;
			
			float maxForce = -FLT_MAX;
			int maxForceIndex = -1;			
			for(auto i=0; i < res.size(); i++){
				vec3 move = 1.0f * vec3(res[i]);
								

				moves[i] += move;

				if (res[i].w > maxForce) {
					maxForce = res[i].w;
					maxForceIndex = i;
				}				
			}

					
		}

		if (moveWallElastic) {
			for (auto i = 0; i < s.size(); i++) {				
				auto & a = s[i];
				const AABB & abb = aabbs[i];
				vec3 F = abb.getContainmentForce(_domain);				
				moves[i] += F;
			}

		}

		//int lowTIndex = -1;		
		if (moveRandomAtLowT && _sa.currentTemperature() < lowT /*&& maxForceIndex != -1*/) {
			

			
		}

		if (_uniform.next() > _sa.currentTemperature()) {
			auto overlap = SDFPerParticleOverlap(s, { _domain.min, _domain.max }, 3);
			//chooseIndexFromVector<float>(overlap, _uniform.next());
			int maxIndex = std::distance(overlap.begin(), std::max_element(overlap.begin(), overlap.end()));
			//lowTIndex = maxIndex;
			moves[maxIndex] = //_domain.min +
				vec3(_normal.next(), _normal.next(), _normal.next()) * _domain.diagonal() * _sa.currentTemperature();
			//std::cout << "lowT move" << std::endl;

		}

		State s1;
		for (auto i = 0; i < s.size(); i++) {
			auto p = s[i];

			assert(!std::isnan(moves[i].x));

			vec3 tvec = -vec3(p.invTransform[3]);
			tvec += moves[i] * (0.1f +  _sa.currentTemperature()) + unattenuatedMoves[i];

			/*if (lowTIndex != -1 && i == lowTIndex) {
				tvec = _domain.min +
					vec3(_uniform.next(), _uniform.next(), _uniform.next()) * _domain.diagonal();
			}*/

			tvec = glm::clamp(tvec, _domain.min, _domain.max);
			p.invTransform[3] = vec4(-tvec, 1.0f);

			//Clamp bounds
			auto tmp = primitiveBounds(p, glm::length(_domain.diagonal()) * 0.5f);;
			AABB bb = { tmp.min, tmp.max };
			vec3 clampMove = bb.getContainment(_domain);
			{
				vec3 tvec = -vec3(p.invTransform[3]);
				tvec += clampMove;
				tvec = glm::clamp(tvec, _domain.min, _domain.max);
				p.invTransform[3] = vec4(-tvec, 1.0f);
			}


			


			s1.push_back(p);
		}
		


		//perturbe state

		return s1;
	};

	_sa.getTemperature = [=](float fraction, size_t iteration){
		const float k = _currentParams.annealingSchedule;
		//const float T0 = 0.05f;
		const float T0 = _currentParams.startingTemperature;
		return T0 * powf(k, float(iteration));
	};

	

	_sa.init(_initState, _currentParams.maxIter);

	return true;
}

bool fast::SDFPacking::step()
{
#ifdef ROTATION_TEST
	auto & s = _sa.state;

	for (auto i = 0; i < s.size(); i++) {		
		mat4 T = glm::inverse(s[i].invTransform);
		vec3 pos = T[3];
		mat4 newT = glm::translate(mat4(1.0f), pos) * glm::rotate(mat4(1.0f), 0.01f, vec3(0, 1, 0)) * glm::translate(mat4(1.0f), -pos) * T;
		s[i].invTransform = glm::inverse(newT);		
	}
	_sa.score(s);
	return true;
#endif

	return _sa.update(1);
}

void fast::SDFPacking::setRasterVolume(Volume * volptr)
{
	_rasterVolume = volptr;	
}

void fast::SDFPacking::rasterize(bool commit)
{
	if (!_rasterVolume) return;


	if (_showBest) {		
		SDFRasterize(_sa.bestState,
		{ _domain.min, _domain.max },
			*_rasterVolume,
			false,
			commit);
		
	}
	else {
		SDFRasterize(_sa.state,
		{ _domain.min, _domain.max },
			*_rasterVolume,
			false,
			commit);
		
	}


}

FAST_EXPORT void fast::SDFPacking::rasterizeOverlap(Volume & volume)
{
	SDFRasterize(_showBest ? _sa.bestState : _sa.state,
	{ _domain.min, _domain.max },
		volume,
		false,
		true, 
		true);
}


std::vector<fast::AABB> fast::SDFPacking::getParticleBounds() const
{
	std::vector<AABB> aabbs;
	auto & s = _sa.state;

	aabbs.resize(s.size(), AABB());
	for (auto i = 0; i < s.size(); i++) {
		auto tmp = primitiveBounds(s[i], glm::length(_domain.diagonal()) * 0.5f);
		aabbs[i] = AABB(tmp.min, tmp.max);
	}

	return aabbs;

}

 float fast::SDFPacking::getMaxPenetrationFraction(int depth, const State * state) const
{
	 const State & s = (state) ? *state : _sa.bestState;
		 

	auto overlap = SDFPerParticleOverlap(s, { _domain.min, _domain.max }, depth);	
	
	int maxIndex = std::distance(overlap.begin(), std::max_element(overlap.begin(), overlap.end()));
	auto & prim = s[maxIndex];
	const std::vector<distfun::Primitive> arr = { prim };
	float volume = SDFVolume(arr, primitiveBounds(prim, glm::length(_domain.diagonal()) * 0.5f), depth + 1);
	float v3 = std::cbrtf(volume);

	float avgOverlap = std::accumulate(overlap.begin(), overlap.end(), 0.0f) / overlap.size();

	//return (overlap[maxIndex] / volume) * v3;
	return (avgOverlap / volume) * v3;

}

 FAST_EXPORT float fast::SDFPacking::getPorosity(int depth /*= 4*/)
 {
	 auto & s = _sa.bestState;
	 float actualVolume = 0.0f;
	 if (_currentParams.allowPartialOutside) {
		 vec3 minUnion = _domain.min - _domain.diagonal()*0.5f;
		 vec3 maxUnion = _domain.max + _domain.diagonal()*0.5f;
		 actualVolume = SDFVolume(s, { minUnion, maxUnion });
	 }
	 else
		 actualVolume = SDFVolume(s, { _domain.min, _domain.max });

	 float porosity = 1.0f - (actualVolume / _domain.volume());
	 return porosity;
 }

 FAST_EXPORT void fast::SDFPacking::setShowBest(bool val)
 {
	 _showBest = val;

	 if (_showBest) {
		 _sa.bestHasChanged = true;
		 rasterize();
	 }
 }
