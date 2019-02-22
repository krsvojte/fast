#pragma once


#include <fastlib/FastLibDef.h>
#include <fastlib/geometry/AABB.h>
#include <fastlib/geometry/Transform.h>

#include <memory>

namespace fast {


	struct Geometry {

		FAST_EXPORT virtual AABB bounds() const = 0;
		FAST_EXPORT virtual std::unique_ptr<Geometry> normalized(bool keepAspectRatio) const = 0;
		FAST_EXPORT virtual std::unique_ptr<Geometry> transformed(const Transform & t) const = 0;
	};

}