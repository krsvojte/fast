#include "utility/PoissonSampler.h"
#include "utility/RandomGenerator.h"

#include <glm/gtx/norm.hpp>

using namespace fast;

//https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
std::vector<fast::vec3> fast::poissonSampler(
	float r, size_t N, AABB domain, 	
	int k /*= 30*/)
{

	RNGUniformInt _uniformInt = RNGUniformInt(0,INT_MAX);
	RNGUniformFloat _uniform = RNGUniformFloat(0, 1.0f);
	_uniformInt.seedByTime();
	_uniform.seedByTime();


	std::vector<vec3> positions;
	
	vec3 diag = domain.diagonal();
	
	const float r2 = r*r;	

	float cellSizeBound = r / glm::sqrt(3);
	ivec3 res = ivec3(glm::ceil(diag.x / cellSizeBound), glm::ceil(diag.y / cellSizeBound), glm::ceil(diag.z / cellSizeBound));
	vec3 cellSize = vec3(diag.x / res.x, diag.y / res.y, diag.z / res.z);
	const ivec3 stride = { 1,res.x, res.x*res.y };

	std::vector<int> grid;
	grid.resize(res.x*res.y*res.z, -1);


	std::vector<int> active;

	auto getIndex = [&](vec3 pos) {
		vec3 posInGrid = (pos - domain.min);
		const ivec3 ipos = vec3(posInGrid.x / cellSize.x, posInGrid.y / cellSize.y, posInGrid.z / cellSize.z);
		return ipos.x + ipos.y * stride.y + ipos.z * stride.z;
	};

	auto getIndexI = [&](const vec3 & ipos) {
		return ipos.x + ipos.y * stride.y + ipos.z * stride.z;
	};



	auto validate = [&](const vec3 & pos) {
		vec3 posInGrid = (pos - domain.min);
		const ivec3 ipos = vec3(posInGrid.x / cellSize.x, posInGrid.y / cellSize.y, posInGrid.z / cellSize.z);
		const size_t index = getIndexI(ipos);

		if (ipos.x < 0 || ipos.y < 0 || ipos.z < 0) return false;
		if (ipos.x >= res.x || ipos.y >= res.y || ipos.z >= res.z) return false;

		if (grid[index] != -1) return false;

		for (auto z = -1; z <= 1; z++) {
			for (auto y = -1; y <= 1; y++) {
				for (auto x = -1; x <= 1; x++) {
					ivec3 nipos = ipos + ivec3(x, y, z);
					if (nipos.x < 0 || nipos.y < 0 || nipos.z < 0) continue;
					if (nipos.x >= res.x || nipos.y >= res.y || nipos.z >= res.z) continue;

					size_t ni = getIndexI(nipos);
					if (grid[ni] != -1 && glm::length2(positions[grid[ni]] - pos) < r2) return false;

				}
			}
		}

		return true;
	};

	vec3 firstPos = domain.min + diag * vec3(_uniform.next(), _uniform.next(), _uniform.next());
	positions.push_back(firstPos);
	grid[getIndex(firstPos)] = 0;
	active.push_back(0);



	while (active.size() > 0 && positions.size() < N) {
		size_t i = _uniformInt.next() % active.size();
		vec3 startPos = positions[i];

		bool added = false;
		for (auto j = 0; j < k; j++) {
			vec3 dir = glm::normalize(vec3(_uniform.next(), _uniform.next(), _uniform.next())) * 2.0f - vec3(1.0f);
			float newR = (_uniform.next()) * r + r; //between r and 2*r
			vec3 newPos = startPos + newR * dir;
			if (validate(newPos)) {
				positions.push_back(newPos);
				grid[getIndex(newPos)] = positions.size() - 1;
				active.push_back(positions.size() - 1);
				added = true;
				break;
			}
		}

		if (!added) {
			active.erase(active.begin() + i);
		}
	}

	//Top up if needed
	while (positions.size() < N) {
		vec3 uniformPos = vec3(_uniform.next(), _uniform.next(), _uniform.next()) * domain.diagonal() + domain.min;
		positions.push_back(uniformPos);
	}


	return positions;

	
}

