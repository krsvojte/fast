#include "geometry/SAP.h"
#include "geometry/GeometryObject.h"

#include <algorithm>


namespace fast {

	FAST_EXPORT void fast::SAP::build(const std::vector<std::shared_ptr<GeometryObject>> & objects, int axis /*= 0*/)
	{
		//Copy and sort
		_array = objects;
		std::sort(_array.begin(), _array.end(), [axis](const ObjPtr & a, const ObjPtr & b) {
			return a->bounds().min[axis] < b->bounds().min[axis];
		});

		//Detect collisions
		sweep();
	}

	FAST_EXPORT std::vector<SAP::CollisionPair> SAP::getCollisionPairs() const
	{
		std::vector<SAP::CollisionPair> objectPairs;
		if (_collisions.size() == 0) return objectPairs;

		objectPairs.reserve(_collisions.size());

		for (auto indexPair : _collisions) {
			objectPairs.push_back(
				{_array[indexPair.first], _array[indexPair.second] }
			);
		}
		return objectPairs;
	}

	void SAP::sweep()
	{
		_collisions.clear();

		int axis = 0;
		auto & arr = _array;

		int curIndices = 0;
		int otherIndices = 1;
		std::vector<size_t> indices[2];


		for (size_t i = 0; i < arr.size(); i++) {
			if (indices[curIndices].size() == 0) {
				indices[curIndices].push_back(i);
				continue;
			}

			float curLeft = arr[i]->bounds().min[axis];

			indices[otherIndices].clear();

			for (auto k : indices[curIndices]) {

				float otherRight = arr[k]->bounds().max[axis];
				if (curLeft > otherRight) {
					//do nothing (remove from next list)						
				}
				else {
					if (arr[i]->bounds().testIntersection(arr[k]->bounds())) {
						_collisions.push_back({
							std::min(i,k), std::max(i,k)
						});
					}
					indices[otherIndices].push_back(k);
				}
			}

			indices[otherIndices].push_back(i);

			std::swap(curIndices, otherIndices);




		}



	}

}