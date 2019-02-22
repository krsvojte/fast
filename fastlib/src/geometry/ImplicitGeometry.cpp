/*
#include "geometry/ImplicitGeometry.h"

#define DISTFUN_IMPLEMENTATION
#include "distfun/distfun.hpp"




namespace fast {

	Particle::Particle()
		: _primitive(std::make_unique<distfun::Primitive>())
	{

	}

	const distfun::Primitive & Particle::getPrimitive() const
	{
		assert(_primitive);
		return *_primitive;
	}

	distfun::Primitive & Particle::getPrimitive()
	{
		assert(_primitive);
		return *_primitive;
	}

}
*/
