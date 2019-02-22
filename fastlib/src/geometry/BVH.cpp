#include "geometry/BVH.h"
//#include "GeometryObject.h"

#ifdef _DEPRECATED
namespace fast {
	FAST_EXPORT void BVH::build(std::vector<std::unique_ptr<GeometryObject>> & objects)
	{		
		
		_root = std::make_unique<Node>();
		_root->_bounds = boundsOfList(objects.begin(), objects.end());


		buildRecursive(_root, objects.begin(), objects.end());

	}

	void BVH::buildRecursive(NodePtr & node, ObjIterator begin, ObjIterator end)
	{
		size_t N = end - begin;


		if (N <= 2) {
			node->objects.resize(N);
			std::move(begin, end, node->objects.begin());
			return;
		}

		int axis = node->_bounds.largestAxis();
		auto cmp = [axis](const ObjPtr & a, const ObjPtr & b) {
			return a->bounds().centroid()[axis] < b->bounds().centroid()[axis];
		};

		std::sort(begin, end, cmp);

		float beginT = (*begin)->bounds().centroid()[axis];
		float endT = (*(begin + N - 1))->bounds().centroid()[axis];
		float midT = (endT - beginT) * 0.5f;

		/*ObjIterator mid = std::lower_bound(begin, end, midT, cmp);
		todo binary search
		*/
		ObjIterator mid = std::find_if(begin, end, [axis, midT](const ObjPtr & a) { return a->bounds().centroid()[axis] > midT; });


		size_t Na = mid - begin;
		size_t Nb = end - mid;

		if (Na == 0 || Nb == 0)
			mid = begin + (N / 2);

		if (Na > 0) {
			auto nodeA = std::make_unique<Node>();
			nodeA->_bounds = boundsOfList(begin, mid);
			buildRecursive(node, begin, mid);
		}

		if (Nb > 0) {
			auto nodeB = std::make_unique<Node>();
			nodeB->_bounds = boundsOfList(mid, end);
			buildRecursive(node, mid, end);
		}




	}

	fast::AABB BVH::boundsOfList(const ObjIterator & begin, const ObjIterator & end)
	{
		AABB bounds;
		for (auto it = begin; it != end; ++it) {
			bounds = bounds.getUnion((*it)->bounds());
		}
		return bounds;
	}

}
#endif