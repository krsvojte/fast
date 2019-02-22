#pragma once

#include "Volume.cuh"

#include "cuda/ManagedAllocator.cuh"




/*
	Outputs labeled volume 
	Returns N - number of labels
	(0 is background, 1...N are labels)
*/
uint VolumeCCL_Label(
	const CUDA_Volume & input,
	CUDA_Volume & output,
	uchar background
);

struct Volume_ComponentIndices {
	/*thrust::device_vector<uint> mask;	*/
	managed_device_vector<uint> origToIndex;
	managed_device_vector<uint> indexToOrig;
	uint NNZ;
	uint3 res;
};

Volume_ComponentIndices VolumeCCL_LinearizeComponents(
	CUDA_Volume & ccl,
	uint numLabels,
	const bool * labelMask
);

/*
	Assigns colors to different labels.
	Outputs uchar4 volume
*/
void VolumeCCL_Colorize(
	const CUDA_Volume & labels, //TYPE_UINT	
	CUDA_Volume & output, //TYPE_UCHAR4
	CUDA_Volume * maskVolume = nullptr, 
	uchar maskVal = 0,
	const bool * maskOptional = nullptr,
	uint numLabelsOptional = 0 
);


/*
	Outputs a bool array of size 6*numLabels 
	[[xpos: label0,label1,label2, ..],[xneg: label0,..]]
*/
void VolumeCCL_BoundaryLabels(
	const CUDA_Volume & labels,
	uint numLabels,
	bool * labelOnBoundary
);

void VolumeCCL_GenerateVolume(
	const CUDA_Volume & labels, //TYPE_UINT
	uint numLabels,
	const bool * labelMask,
	CUDA_Volume & output //TYPE_UCHAR
);