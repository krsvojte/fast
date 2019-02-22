#include "LinearSystemDevice.cuh"
#include <stdlib.h>
#include "cuda/CudaUtility.h"
using namespace fast;


template <typename T>
__global__ void ___generateDomain(
	const CUDA_Volume binaryMask,
	double value_zero,
	double value_one,
	CUDA_Volume output
) {
	VOLUME_VOX_GUARD(output.res);

	//Read mask
	uchar c = read<uchar>(binaryMask.surf, vox);

	//Write value
	write<T>(output.surf, vox, (c > 0) ? T(value_one) : T(value_zero));
}


void fast::GenerateDomain(const CUDA_Volume & binaryMask, double value_zero, double value_one, CUDA_Volume & output)
{
	BLOCKS3D(8, output.res);

	if (binaryMask.type != TYPE_UCHAR) {
		exit(1);
	}

	if (output.type == TYPE_DOUBLE) {
		___generateDomain<double> << < numBlocks, block >> > (
			binaryMask,
			value_zero,
			value_one,
			output
			);
	}
	else if (output.type == TYPE_FLOAT) {
		___generateDomain<float> << < numBlocks, block >> > (
			binaryMask,
			value_zero,
			value_one,
			output
			);
	}
	else {
		exit(2);
	}

#ifdef _DEBUG
	_CUDA(cudaDeviceSynchronize());
#endif

	
}
