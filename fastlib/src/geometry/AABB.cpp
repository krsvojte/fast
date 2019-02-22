#include "geometry/AABB.h"

using namespace fast;

vec3 fast::AABB::getContainment(const AABB & container) const
{
	vec3 posDiff = max - container.max;
	posDiff = glm::max(vec3(0.0f), posDiff);	

	vec3 negDiff = container.min - min;
	negDiff = glm::max(vec3(0.0f), negDiff);

	return -(posDiff - negDiff);
}

vec3 fast::AABB::getContainmentForce(const AABB & container) const
{
	
	vec3 posDiff = max - container.max;
	posDiff = glm::max(vec3(0.0f), posDiff);
	posDiff *= posDiff;

	vec3 negDiff = container.min - min;
	negDiff = glm::max(vec3(0.0f), negDiff);
	negDiff *= negDiff;


	return -(posDiff - negDiff);

}

