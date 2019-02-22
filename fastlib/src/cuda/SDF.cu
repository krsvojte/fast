#include "SDF.cuh"
#include "cuda/ManagedAllocator.cuh"


using namespace fast;


distfun::vec4 fast::launchElasticityKernel(
	const std::vector<PrimitiveCollisionPair> & pairs
)
{
	managed_device_vector<PrimitiveCollisionPair> d_pairs = pairs;


	/*char b;
	b = 0;*/
	//Per voxel variant

	/*
		TODO
	*/



	return { 0,0,0,0 };
}