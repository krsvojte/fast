#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/Types.h>

namespace fast {

	class LinearSystem {		
		
	public:
		virtual ~LinearSystem() {}
		virtual PrimitiveType type() = 0;
		ivec3 res;
		uint NNZ;
	};	

}