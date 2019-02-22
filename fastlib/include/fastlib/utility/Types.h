#pragma once

#pragma warning(push, 0) 
#include <glm/glm.hpp>
#pragma warning(pop)
#include "PrimitiveTypes.h"

namespace fast {

	using vec2 = glm::vec2;
	using vec3 = glm::vec3;
	using vec4 = glm::vec4;

	using color3 = glm::vec3;
	using color4 = glm::vec4;

	using ivec2 = glm::ivec2;
	using ivec3 = glm::ivec3;
	using ivec4 = glm::ivec4;

	using mat2 = glm::mat2;
	using mat3 = glm::mat3;
	using mat4 = glm::mat4;


	inline size_t linearIndex(const ivec3 & dim, int x, int y, int z) {
		return x + dim.x * y + dim.x * dim.y * z;
	}

	inline size_t linearIndex(const ivec3 & dim, const ivec3 & pos) {
		return pos.x + dim.x * pos.y + dim.x * dim.y * pos.z;
	}

	inline bool isValidPos(const ivec3 & dim, const ivec3 & pos) {
		return	pos.x >= 0 && pos.y >= 0 && pos.z >= 0 &&
				pos.x < dim.x && pos.y < dim.y && pos.z < dim.z;
	}

	inline ivec3 posFromLinear(const ivec3 & dim, size_t index) {

		ivec3 pos;
		pos.x = static_cast<int>(index % dim.x);
		index = (index - pos.x) / dim.x;
		pos.y = static_cast<int>(index % dim.y);
		pos.z = static_cast<int>(index / dim.y);
		return pos;

		//return pos.x + dim.x * pos.y + dim.x * dim.y * pos.z;
	}

}