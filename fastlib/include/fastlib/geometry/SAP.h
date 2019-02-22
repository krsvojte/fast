#pragma once

#include <fastlib/FastLibDef.h>


#include <memory>
#include <vector>

namespace fast {

	struct GeometryObject;

	class SAP {
		
		using ObjPtr = std::shared_ptr<GeometryObject>;
		using ObjArr = std::vector<ObjPtr>;
		using CollisionPair = std::pair<ObjPtr, ObjPtr>;
		

	public:
		FAST_EXPORT void build(const std::vector<std::shared_ptr<GeometryObject>> & objects, int axis = 0);
		
		FAST_EXPORT std::vector<CollisionPair> getCollisionPairs() const;
		
	private:
		using CollisionPairIndices = std::pair<size_t, size_t>;
		std::vector<CollisionPairIndices> _collisions;
		ObjArr _array;

		void sweep();

	};

}