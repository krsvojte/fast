#pragma once

#include <fastlib/utility/Types.h>

namespace fast {
	

	struct AABB {
		vec3 min;// = vec3(FLT_MAX);
		vec3 max;// = vec3(-FLT_MAX);

		AABB(vec3 vmin = vec3(FLT_MAX), vec3 vmax = vec3(-FLT_MAX)) : min(vmin),max(vmax) {}

		static AABB unit() {
			return AABB(vec3(0), vec3(1));
		}

		AABB getUnion(const AABB & b) {
			AABB c;
			c.min = glm::min(min, b.min);
			c.max = glm::max(max, b.max);
			return c;
		}

		bool contains(const AABB & b) {
			return (min.x <= b.min.x && min.y <= b.min.y && min.z <= b.min.z
					&& max.x >=  b.max.x && max.y >= b.max.y && max.z >= b.max.z);
		}

		vec3 range() const {
			return max - min;
		}

		vec3 centroid() const {
			return min + range() * 0.5f;
		}

		AABB getIntersection(const AABB & b) const {
			return { glm::max(min, b.min), glm::min(max, b.max) };
		}

		bool isValid() const
		{
			return min.x < max.x && min.y < max.y && min.z < max.z;
		}

		bool testIntersection(const AABB & b) const {
			return getIntersection(b).isValid();
		}

		vec3 diagonal() const {
			return max - min;
		}

		float volume() const {
			auto diag = diagonal();
			return diag.x * diag.y * diag.z;
		}

		int largestAxis() const {
			vec3 r = range();
			if (r.x > r.y) {
				if (r.x > r.z) return 0;
				return 2;
			}			
			if (r.y > r.z) return 1;
			return 2;						
		}

		vec3 getContainment(const AABB & container) const;
		vec3 getContainmentForce(const AABB & container) const;

		vec3 center() const {
			return (min + max) * 0.5f;
		}
	};
}