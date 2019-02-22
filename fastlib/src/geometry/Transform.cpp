#include "geometry/Transform.h"

#include <glm/gtc/matrix_transform.hpp>

namespace fast {

	FAST_EXPORT mat4 fast::Transform::getAffine() const
	{
		return glm::translate(glm::mat4(1.0f), translation) * getRotation<mat4>() * glm::scale(glm::mat4(1.0f), scale);
	}

	FAST_EXPORT mat4 fast::Transform::getInverseAffine() const
	{
		return 
			glm::scale(glm::mat4(1.0f), vec3(1.0f / scale.x, 1.0f / scale.y, 1.0f / scale.z)) *
			getInverseRotation<mat4>() * 
			glm::translate(glm::mat4(1.0f), -translation);
	}

	
	FAST_EXPORT vec3 Transform::transformPoint(const vec3 & pt) const
	{
		return vec3(getAffine() * vec4(pt, 1.0f));
	}

	FAST_EXPORT vec3 Transform::transformVector(const vec3 & vec) const
	{
		return vec3(getAffine() * vec4(vec, 0.0f));
	}

}


/*

using namespace Eigen;

Affine3f fast::EigenTransform::getAffine() const
{
	return Translation3f(translation) * rotation * Scaling(scale[0], scale[1], scale[2]);
}

Affine3f fast::EigenTransform::getInverseAffine() const
{
	//todo: use conjugate quat?
	return 
		Scaling(1.0f / scale[0],  1.0f / scale[1], 1.0f / scale[2]) 
		*  rotation.inverse() 
		* Translation3f(-translation);
}

Eigen::Matrix3f fast::EigenTransform::getRotation() const
{
	return rotation.toRotationMatrix();
}

Eigen::Vector3f fast::EigenTransform::applyToPointInverse(const Eigen::Vector3f & point) const
{
	return getInverseAffine() * point;
}

*/
