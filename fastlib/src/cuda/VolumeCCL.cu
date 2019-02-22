#include "VolumeCCL.cuh"
#include <stdio.h>
#include <assert.h>


#include <thrust/reduce.h>
#include <set>
#include <stdlib.h>

/*
	Based on
	Ohira, N. (2018). Memory-efficient 3D connected component labeling with parallel computing. Signal, Image and Video Processing, 12(3), 429-436.
*/


struct VolumeCCL_Params {
	size_t spanCount;

	//Input resolution	
	uint3 res;

	//Count of spans for each row in y-z plane
	uint * spanCountPlane;

	//Resolution of span/label volumes
	uint3 spanRes;	
	uint2 * spanVolume;
	uint * labelVolume;
	
	//Buffer for inclusive scan of labels
	uint * labelScan;

	bool * hasChanged;
};

//https://en.wikipedia.org/wiki/Help:Distinguishable_colors
#define COLOR_N 26
__device__ __constant__ uchar3 const_colors[COLOR_N];

#define EMPTY_COLOR make_uchar3(0,0,0)

uchar3 colors_CAP[COLOR_N] = {	
	{ 255,80,5 },
	{ 0,117,220 }, //blue
	{ 43,206,72 }, //green
	{ 255,0,16 }, //red
	{ 240,163,255 },	
	{ 153,63,0 },
	{ 76,0,92 },
	{ 25,25,25 },
	{ 0,92,49 },	
	{ 255,204,153 },
	{ 128,128,128 },
	{ 148,255,181 },
	{ 143,124,0 },
	{ 157,204,0 },
	{ 194,0,136 },
	{ 0,51,128 },
	{ 255,164,5 },
	{ 255,168,187 },
	{ 66,102,0 },	
	{ 94,241,242 },
	{ 0,153,143 },
	{ 224,255,102 },
	{ 116,10,255 },
	{ 153,0,0 },
	{ 255,255,128 },
	{ 255,255,0 }	
};



template <typename T>
inline __device__ bool opEqual(const T & a, const T &  b, const T & threshold = 0) {
	return (a == b);
}


template <typename T>
using CmpOp = bool(*)(
	const T & a, const T & b, const T & threshold
	);



template <typename T, CmpOp<T> _cmpOp>
__global__ void __countSpansKernel(
	CUDA_Volume input, 
	uint * result,
	T background,
	T threshold
) {

	const uint3 resultRes = make_uint3(1, input.res.y, input.res.z);
	VOLUME_IVOX_GUARD(resultRes);
	

	uint spanSum = 0;
	int3 curPos = ivox;

	T prevVal = background;

	bool open = false;
	for (int i = 0; i < input.res.x; i++) {
		T newVal = read<T>(input.surf, curPos);
		
		if (!_cmpOp(prevVal, newVal, threshold)) {
			open = !open;
			if (open) {
				spanSum++;				
			}
		}		
		prevVal = newVal;
		curPos.x++;
	}

	size_t linIndex = _linearIndex(resultRes, ivox);
	result[linIndex] = spanSum;

}


/*
	Detect spans again and assign them to span and label matrices
*/
template <typename T, CmpOp<T> _cmpOp>
__global__ void __buildMatricesKernel(
	CUDA_Volume input,
	VolumeCCL_Params p,
	T background,
	T threshold
) {
	const uint3 planeRes = make_uint3(1, input.res.y, input.res.z);
	VOLUME_IVOX_GUARD(planeRes);	

	uint currentSpanIndex = 0;
	int3 curPos = ivox;

	T prevVal = background;
	uint spanBegin = 0;

	bool open = false;
	for (int i = 0; i < input.res.x; i++) {
		T newVal = read<T>(input.surf, curPos);

		if (!_cmpOp(prevVal,newVal,threshold)) {
			if (!open) {				
				spanBegin = i;								
			}
			else {		
				size_t index = _linearIndex(p.spanRes, make_int3(currentSpanIndex, ivox.y, ivox.z));
				p.spanVolume[index] = make_uint2(spanBegin, i - 1);				
				p.labelVolume[index] = _linearIndex(input.res, make_uint3(spanBegin, ivox.y, ivox.z));
				currentSpanIndex++;
			}
			open = !open;
		}
		prevVal = newVal;
		curPos.x++;
	}

	//Close last one if open
	if (open) {
		size_t index = _linearIndex(p.spanRes, make_int3(currentSpanIndex, ivox.y, ivox.z));
		p.spanVolume[index] = make_uint2(spanBegin, input.res.x - 1);
		p.labelVolume[index] = _linearIndex(input.res, make_uint3(spanBegin, ivox.y, ivox.z));
	}

}


inline __device__ bool spanOverlap(uint2 a, uint2 b) {
	return a.x <= b.y && b.x <= a.y;
}

inline __device__ uint labelEquivalence(VolumeCCL_Params & p, uint index, int3 ivox) {

	
	
	uint indexPrev = 0;	
	do {
		indexPrev = index;	

		//Find pos of label
		uint3 posOrig = posFromLinear(p.res, index);
		uint3 posLabel = make_uint3(0, posOrig.y, posOrig.z);

		const uint rowSpanCount = p.spanCountPlane[_linearIndex(make_uint3(1, p.res.y, p.res.z), posLabel)];
		for (int i = 0; i < rowSpanCount; i++) {			
			uint2 span = p.spanVolume[_linearIndex(p.spanRes, posLabel)];
			if (posOrig.x >= span.x && posOrig.x <= span.y)
				break;		
			posLabel.x++;
		}
		index = p.labelVolume[_linearIndex(p.spanRes, posLabel)];
		
	}
	while(index != indexPrev);

	return index;
}

//input surface not needed
__global__ void __updateContinuityKernel(VolumeCCL_Params p) {
	const uint3 planeRes = make_uint3(1, p.res.y, p.res.z);
	VOLUME_IVOX_GUARD(planeRes);

	const uint rowSpanCount = p.spanCountPlane[_linearIndex(planeRes, ivox)];
	
	int3 curPosSpan = ivox;
	

	const int3 offsets[4] = {
		{ 0,-1,0 },
		{ 0,1,0 },
		{ 0,0,-1 },
		{ 0,0,1 },
	};

	int3 otherPosSpan[4] = {
		ivox + offsets[0], ivox + offsets[1], ivox + offsets[2], ivox + offsets[3]
	};



	const uint otherSpanCount[4] = {
		(ivox.y == 0) ?				0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[0])],
		(ivox.y == p.res.y - 1) ?	0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[1])],
		(ivox.z == 0) ?				0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[2])],
		(ivox.z == p.res.z - 1) ?	0 : p.spanCountPlane[_linearIndex(planeRes, ivox + offsets[3])]
	};

	for (int i = 0; i < rowSpanCount; i++) {
		uint2 thisSpan = p.spanVolume[_linearIndex(p.spanRes, curPosSpan)];		
		uint tempLabel = p.labelVolume[_linearIndex(p.spanRes, curPosSpan)];
		

		

		#pragma unroll
		for (int k = 0; k < 4; k++) {
			

			if (k == 0 && ivox.y == 0) continue;
			if (k == 1 && ivox.y == p.res.y - 1) continue;
			if (k == 2 && ivox.z == 0) continue;
			if (k == 3 && ivox.z == p.res.z - 1) continue;
	
			
			while(otherPosSpan[k].x < otherSpanCount[k]){
				uint2 otherSpan = p.spanVolume[_linearIndex(p.spanRes, otherPosSpan[k])];

				

				if (otherSpan.x > thisSpan.y) break;							

				if (spanOverlap(thisSpan, otherSpan)) {										
					uint * thisLabelPtr = p.labelVolume + _linearIndex(p.spanRes, curPosSpan);
					uint * otherLabelPtr = p.labelVolume + _linearIndex(p.spanRes, otherPosSpan[k]);

					uint thisLabel = *thisLabelPtr;
					uint otherLabel = *otherLabelPtr;
					
					if (thisLabel < otherLabel) {
						atomicMin(otherLabelPtr, thisLabel);
						*p.hasChanged = true;
					}
					else if(otherLabel < thisLabel) {
						atomicMin(thisLabelPtr, otherLabel);
						*p.hasChanged = true;
					}				
					
				}

				otherPosSpan[k].x++;				
			}
		}	

		curPosSpan.x++;
		

	}


	curPosSpan = ivox;	
	size_t index0 = _linearIndex(p.res, ivox);	

	for (int i = 0; i < rowSpanCount; i++) {
		uint index = p.labelVolume[_linearIndex(p.spanRes, curPosSpan)];


		p.labelVolume[_linearIndex(p.spanRes, curPosSpan)] = labelEquivalence(p, index, ivox);

		curPosSpan.x++;		
	}


}



__global__ void __labelOutputKernel(
	VolumeCCL_Params p,
	CUDA_Volume output
	
) {
	const uint3 planeRes = make_uint3(1, p.res.y, p.res.z);
	VOLUME_IVOX_GUARD(planeRes);

	const uint rowSpanCount = p.spanCountPlane[_linearIndex(planeRes, ivox)];

	int3 posSpan = ivox;
	for (; posSpan.x < rowSpanCount; posSpan.x++) {

		const uint2 thisSpan = p.spanVolume[_linearIndex(p.spanRes, posSpan)];
		const uint thisLabel = p.labelVolume[_linearIndex(p.spanRes, posSpan)];
		
		for (int k = thisSpan.x; k <= thisSpan.y; k++) {
			const int3 pos = make_int3(k, ivox.y, ivox.z);
			write<uint>(output.surf, pos, thisLabel);
		}		
	}

}

__global__ void __markRootLabelsKernel(VolumeCCL_Params p){	
	const uint3 planeRes = make_uint3(1, p.res.y, p.res.z);
	VOLUME_IVOX_GUARD(planeRes);

	const uint rowSpanCount = p.spanCountPlane[_linearIndex(planeRes, ivox)];

	int3 posSpan = ivox;
	for (; posSpan.x < rowSpanCount; posSpan.x++) {
		const size_t spanIndex = _linearIndex(p.spanRes, posSpan);
		const uint2 thisSpan = p.spanVolume[spanIndex];
		const uint thisLabel = p.labelVolume[spanIndex];

		const uint spanBeginIndex = _linearIndex(p.res, ivox + make_int3(thisSpan.x, 0, 0));		
		p.labelScan[spanIndex] = (spanBeginIndex == thisLabel) ? 1 : 0;		
	}

}

__global__ void __reindexLabels(VolumeCCL_Params p) {
	const uint3 planeRes = make_uint3(1, p.res.y, p.res.z);
	VOLUME_IVOX_GUARD(planeRes);

	const uint rowSpanCount = p.spanCountPlane[_linearIndex(planeRes, ivox)];

	int3 posSpan = ivox;
	for (; posSpan.x < rowSpanCount; posSpan.x++) {
		const size_t spanIndex = _linearIndex(p.spanRes, posSpan);
		//const uint2 thisSpan = p.spanVolume[spanIndex];
		const uint thisLabel = p.labelVolume[spanIndex];
		
		//Find original label row
		uint3 origLabelPos = posFromLinear(p.res, thisLabel);
		const uint origXPos = origLabelPos.x;
		origLabelPos.x = 0;
		
		uint newLabel = uint(-1);
		//Iterate through the row to find original index (based on span begin)
		const uint otherRowSpanCount = p.spanCountPlane[_linearIndex(planeRes, origLabelPos)];
		for (; origLabelPos.x < otherRowSpanCount; origLabelPos.x++){
			size_t otherIndex = _linearIndex(p.spanRes, origLabelPos);
			//When span found, get scanned label (new unique label) 
			if (origXPos == p.spanVolume[otherIndex].x) {
				newLabel = p.labelScan[otherIndex];
			}
		}

		p.labelVolume[spanIndex] = newLabel;		
	}

}

#ifdef _DEBUG 
#define DEBUG_CPU
#endif

uint VolumeCCL_Label(const CUDA_Volume & input, CUDA_Volume & output, uchar background)
{
	assert(input.type == TYPE_UCHAR);
	
	const uint YZBlockSize = 32;

	
	VolumeCCL_Params p;	
	p.res = input.res;

	uint numLabels = 0;

	managed_device_vector<uint> sumPlane(input.res.y * input.res.z);
	p.spanCountPlane = sumPlane.data().get();

	//Count spands
	{
		BLOCKS3D_INT3(1, YZBlockSize, YZBlockSize, make_uint3(1, input.res.y, input.res.z));
		//Summing in ascending X direction
		
		if (input.type == TYPE_UCHAR) {			
			uchar threshold = 1;
			__countSpansKernel<uchar, opEqual<uchar>> << < numBlocks, block >> > (input, p.spanCountPlane, background, threshold);
		}
		else {
			assert("Unsupported type");
			exit(0);
		}
		uint maxSpanCount = thrust::reduce(sumPlane.begin(), sumPlane.end(), 0, thrust::maximum<uint>());


		p.spanCount = maxSpanCount;
		p.spanRes = make_uint3(p.spanCount, p.res.y, p.res.z);		
		
	}
#ifdef DEBUG_CPU
	thrust::host_vector<uint> hostSum = sumPlane;
	uint * dataSum = hostSum.data();
#endif
	

	{
		

		managed_device_vector<uint2> spanMatrix(p.spanRes.x * p.spanRes.y * p.spanRes.z);
		managed_device_vector<uint> labelMatrix(p.spanRes.x * p.spanRes.y * p.spanRes.z);
		managed_device_vector<uint> labelScan(p.spanRes.x * p.spanRes.y * p.spanRes.z);

		p.spanVolume = spanMatrix.data().get();
		p.labelVolume = labelMatrix.data().get();
		p.labelScan = labelScan.data().get();

#ifdef DEBUG_CPU
		thrust::host_vector<uint2> hostSpan;
		thrust::host_vector<uint> hostLabel;
		thrust::host_vector<uint> hostLabelScan;

		cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 64);
#endif

		cudaMalloc(&p.hasChanged, 1);
		cudaMemset(p.hasChanged, 0, 1);

		{
			BLOCKS3D_INT3(1, YZBlockSize, YZBlockSize, make_uint3(1, input.res.y, input.res.z));
			if (input.type == TYPE_UCHAR) {				
				uchar threshold = 1;				
				__buildMatricesKernel<uchar, opEqual<uchar>> << < numBlocks, block >> > (input, p, background, threshold);			
			}
			else {
				assert("Unsupported type");
				exit(0);
			}
			
		}

#ifdef DEBUG_CPU
		{
			hostSpan = spanMatrix;  uint2 * dataSpan = hostSpan.data();
			hostLabel = labelMatrix; uint * dataLabel = hostLabel.data();
			char b;
			b = 0;
		}

		
#endif
		std::set<uint> uniqueLabels;

		bool hasChangedHost = false;
#ifdef DEBUG_CPU
		int iteration = 0;
#endif
		do
		{
			BLOCKS3D_INT3(1, YZBlockSize, YZBlockSize, make_uint3(1, input.res.y, input.res.z));
			__updateContinuityKernel << < numBlocks, block >> >(p);
			cudaMemcpy(&hasChangedHost, p.hasChanged, 1, cudaMemcpyDeviceToHost);
			cudaMemset(p.hasChanged, 0, 1);

#ifdef DEBUG_CPU
			{
								
				hostSpan = spanMatrix;  uint2 * dataSpan = hostSpan.data();
				hostLabel = labelMatrix; uint * dataLabel = hostLabel.data();
				
				uniqueLabels.clear();
				for (auto i = 0; i < p.spanRes.x * p.spanRes.y * p.spanRes.z; i++) {
					uniqueLabels.insert(hostLabel[i]);
				}				

				
				char b;
				b = 0;

				iteration++;
			}
#endif

		} while (hasChangedHost);		
		cudaFree(p.hasChanged);


#ifdef DEBUG_CPU
		bool isZeroALabel = false;
		{
			uint firstCount;
			uint firstLabel;
			uint2 firstSpan;

			cudaMemcpy(&firstCount, p.spanCountPlane, sizeof(uint), cudaMemcpyDeviceToHost);
			cudaMemcpy(&firstLabel, p.labelVolume, sizeof(uint), cudaMemcpyDeviceToHost);
			cudaMemcpy(&firstSpan, p.spanVolume, sizeof(uint2), cudaMemcpyDeviceToHost);

			if (firstCount > 0 && firstLabel == 0 && firstSpan.x == 0) {
				isZeroALabel = true;
			}
		}
		
		if (!isZeroALabel) { 
			uniqueLabels.erase(0);
		}
		
		for (auto & s : uniqueLabels) {
			printf("%u, ", s);
		}
		printf("\n");
#endif

		


		//reindexing
		{
			//label matrix where first labeled (label == data idnex) is marked as 1, otherwise 0
			{
				BLOCKS3D_INT3(1, YZBlockSize, YZBlockSize, make_uint3(1, input.res.y, input.res.z));
				__markRootLabelsKernel << < numBlocks, block >> > (p);

#ifdef DEBUG_CPU
				hostLabelScan = labelScan;
				uint * dataLabelScan = hostLabelScan.data();				
#endif
			}

			//inclusive scan			
			{
				thrust::inclusive_scan(labelScan.begin(), labelScan.end(), labelScan.begin());

#ifdef DEBUG_CPU
				hostLabelScan = labelScan;
				uint * dataLabelScan = hostLabelScan.data();
				char b;
				b = 0;
#endif
			}
			//get number of unique labels
			
			{
				cudaMemcpy(&numLabels, p.labelScan + p.spanRes.x * p.spanRes.y * p.spanRes.z - 1, sizeof(uint), cudaMemcpyDeviceToHost);
				numLabels += 1; //Include 0 label -> not labeled
			}

			//printf("Label count: %u\n", numLabels);			
			
			//reindex
			{
				BLOCKS3D_INT3(1, YZBlockSize, YZBlockSize, make_uint3(1, input.res.y, input.res.z));
				__reindexLabels << < numBlocks, block >> > (p);

#ifdef DEBUG_CPU
				{

					hostSpan = spanMatrix;  uint2 * dataSpan = hostSpan.data();
					hostLabel = labelMatrix; uint * dataLabel = hostLabel.data();

					uniqueLabels.clear();
					for (auto i = 0; i < p.spanRes.x * p.spanRes.y * p.spanRes.z; i++) {
						uniqueLabels.insert(hostLabel[i]);
					}

					for (auto & s : uniqueLabels) {
						printf("%u, ", s);
					}
					printf("\n");

				}
#endif
			}

		}

		//Reconstruct volume
		/*{
			assert(output.type == TYPE_UCHAR);
			BLOCKS3D_INT3(1, 8, 8, make_uint3(1, input.res.y, input.res.z));
			__labelOutputKernelUchar << < numBlocks, block >> > (p, output, label);
		}*/

		{
			assert(output.type == TYPE_UINT);
			BLOCKS3D_INT3(1, YZBlockSize, YZBlockSize, make_uint3(1, input.res.y, input.res.z));
			__labelOutputKernel << < numBlocks, block >> > (p, output);
		}

		//Need outputs:
		/*
			a) uint, 0 = background, 1..N labels (for storage/representation) | done
			b) float3, colormap/randomized -> outside of this function (for visualization) | done
			c) uchar, 0 none, 1 selected label(s) -> for reactive area density / filtering | todo:
				need to know labels in a boundary slice
		*/
		

	}

	


	return numLabels;
}


__global__ void __colorizeKernel(CUDA_Volume input, CUDA_Volume output) {
	
	VOLUME_IVOX_GUARD(input.res);

	uint label = read<uint>(input.surf, ivox);
	if (label != 0) {
		size_t colorIndex = label % COLOR_N;
		uchar3 rgb = const_colors[colorIndex];
		uchar4 color = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
		write<uchar4>(output.surf, ivox, color);
	}
	else {
		write<uchar4>(output.surf, ivox, make_uchar4(EMPTY_COLOR.x, EMPTY_COLOR.y, EMPTY_COLOR.z,0));
	}

}

__global__ void __colorizeMaskedLabelsKernel(CUDA_Volume input, CUDA_Volume output, bool * mask) {

	VOLUME_IVOX_GUARD(input.res);

	uint label = read<uint>(input.surf, ivox);
	if (!mask[label])
		return;

	if (label != 0) {
		size_t colorIndex = label % COLOR_N;
		uchar3 rgb = const_colors[colorIndex];
		uchar4 color = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
		write<uchar4>(output.surf, ivox, color);
	}
	else {
		write<uchar4>(output.surf, ivox, make_uchar4(EMPTY_COLOR.x, EMPTY_COLOR.y, EMPTY_COLOR.z, 0));
	}
}

__global__ void __colorizeMaskedVolumeKernel(CUDA_Volume input, CUDA_Volume output, CUDA_Volume mask, uchar maskVal) {

	VOLUME_IVOX_GUARD(input.res);

	if (read<uchar>(mask.surf, ivox) != maskVal)
		return;

	uint label = read<uint>(input.surf, ivox);
	if (label != 0) {
		size_t colorIndex = label % COLOR_N;
		uchar3 rgb = const_colors[colorIndex];
		uchar4 color = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
		write<uchar4>(output.surf, ivox, color);
	}
	else {
		write<uchar4>(output.surf, ivox, make_uchar4(EMPTY_COLOR.x, EMPTY_COLOR.y, EMPTY_COLOR.z, 0));
	}
}

void VolumeCCL_Colorize(
	const CUDA_Volume & input, 
	CUDA_Volume & output, 
	CUDA_Volume * maskVolume,
	uchar maskVal,
	const bool * maskOptional, 
	uint numLabelsOptional
)
{
	
	cudaError_t res = cudaMemcpyToSymbol(
		const_colors,
		&colors_CAP,
		sizeof(uchar3) * COLOR_N,
		0,
		cudaMemcpyHostToDevice
	);
	assert(res == cudaSuccess);



	assert(input.type == TYPE_UINT);
	assert(output.type == TYPE_UCHAR4);


	BLOCKS3D(8, input.res);

	if (maskVolume) {
		__colorizeMaskedVolumeKernel << < numBlocks, block >> > (input, output, *maskVolume, maskVal);
	}
	else if (maskOptional && numLabelsOptional > 0) {
		
		bool * maskDevice;
		{
			size_t maskBytesize = numLabelsOptional * sizeof(bool);
			cudaMalloc(&maskDevice, maskBytesize);
			cudaMemset(maskDevice, 0, maskBytesize);
			cudaMemcpy(maskDevice, maskDevice, maskBytesize, cudaMemcpyHostToDevice);
		}

		__colorizeMaskedLabelsKernel << < numBlocks, block >> > (input, output, maskDevice);

		{
			cudaFree(maskDevice);
		}

	}
	else {
		__colorizeKernel << < numBlocks, block >> > (input, output);
	}

	

}



__global__ void ___boundaryLabelsKernel(
	CUDA_Volume input,
	uint3 res, 
	uint3 offset,
	bool * labelBitmap
){	
	VOLUME_VOX_GUARD(res);
	vox += offset;

	uint label = read<uint>(input.surf, vox);

	if (label > 0) {
		labelBitmap[label] = true;
	}

}


void VolumeCCL_BoundaryLabels(const CUDA_Volume & labels, uint numLabels, bool * labelOnBoundary)
{

	assert(labels.type == TYPE_UINT);
	assert(labelOnBoundary != nullptr);

	
	size_t boundaries = 6;	

	bool * boolmap;
	size_t boolmapBytesize = boundaries * numLabels * sizeof(bool);
	cudaMalloc(&boolmap, boolmapBytesize);
	cudaMemset(boolmap, 0, boolmapBytesize);
	

	for (int i = 0; i < DIR_NONE; i++) {	
		Dir dir = Dir(i);

		int primaryDir = getDirIndex(dir);
		int secondaryDirs[2] = { (primaryDir + 1) % 3, (primaryDir + 2) % 3 };

		uint _res[3];
		_res[primaryDir] = 1;
		_res[secondaryDirs[0]] = ((uint*)&labels.res)[secondaryDirs[0]];
		_res[secondaryDirs[1]] = ((uint*)&labels.res)[secondaryDirs[1]];

		uint3 res = make_uint3(_res[0], _res[1], _res[2]);

		uint _block[3];
		_block[primaryDir] = 1;
		_block[secondaryDirs[0]] = 32;
		_block[secondaryDirs[1]] = 32;

		uint3 offset = make_uint3(0);
		if (getDirSgn(dir) == 1) {
			((uint*)&offset)[primaryDir] = ((uint*)&labels.res)[primaryDir] - 1;
		}

		//printf("dir %d, %u %u %u\n", i, res[0], res[1], res[2]);
		BLOCKS3D_INT3(_block[0],_block[1],_block[2], res);

		___boundaryLabelsKernel << < numBlocks, block >> > (
			labels, 
			res,
			offset,
			boolmap + i*numLabels
		);
	}

	cudaMemcpy(labelOnBoundary, boolmap, boolmapBytesize, cudaMemcpyDeviceToHost);
	cudaFree(boolmap);

	//Add Dir none (accessible from all)
	bool * arrAll = labelOnBoundary + int(DIR_NONE)*numLabels;
	memset(arrAll, 1, numLabels * sizeof(bool));

	for (auto i = 0; i < DIR_NONE; i++) {
		bool * arr = labelOnBoundary + i*numLabels;		
		for (auto j = 0; j < numLabels; j++) {
			arrAll[j] &= arr[j];
		}
	}

	return;

}


__global__ void __generateVolumeFromLabelsKernel(
	CUDA_Volume labels,
	bool * mask,
	CUDA_Volume output
) {
	VOLUME_IVOX_GUARD(labels.res);

	uint label = read<uint>(labels.surf, ivox);
	if (mask[label]) {
		write<uchar>(output.surf, ivox, 255);
	}
}


void VolumeCCL_GenerateVolume(
	const CUDA_Volume & labels,
	uint numLabels,
	const bool * labelMask, 
	CUDA_Volume & output )
{
	assert(labels.type == TYPE_UINT);
	assert(output.type == TYPE_UCHAR);
	assert(labelMask != nullptr);

	size_t maskBytesize = numLabels * sizeof(bool);
	bool * maskDevice;
	cudaMalloc(&maskDevice, maskBytesize);
	cudaMemset(maskDevice, 0, maskBytesize);
	cudaMemcpy(maskDevice, labelMask, maskBytesize, cudaMemcpyHostToDevice);


	{
		BLOCKS3D(8, labels.res);
		__generateVolumeFromLabelsKernel << < numBlocks, block >> > (labels, maskDevice, output);
	}


	cudaFree(maskDevice);

}


void __global__ __maskCCLKernel(const CUDA_Volume ccl, const bool * labelMask, uint * mask) {
	VOLUME_IVOX_GUARD(ccl.res);

	uint label = read<uint>(ccl.surf, ivox);
	size_t i = _linearIndex(ccl.res, ivox);
	if (labelMask[label]) {
		mask[i] = 1;
	}
	else {
		mask[i] = 0;
	}
}

void __global__ __indexToOrigKernel(
	uint3 res,
	uint * mask,
	uint * indices,
	uint * indexToOrig
){
	VOLUME_IVOX_GUARD(res);

	size_t indexOrig = _linearIndex(res, ivox);
	if (mask[indexOrig]) {
		uint newIndex = indices[indexOrig];
		indexToOrig[newIndex] = indexOrig;
	}
	else {
		indices[indexOrig] = uint(-1);
	}

}

template <typename T>
T _scanAndCount(const managed_device_vector<T> & input, managed_device_vector<T> & scanned) {
	thrust::exclusive_scan(input.begin(),
		input.end(),
		scanned.begin());

	
	uint lastInput = input.back();
	uint lastScan = scanned.back(); 	
	return lastInput + lastScan;
}


Volume_ComponentIndices VolumeCCL_LinearizeComponents(CUDA_Volume & ccl, uint numLabels, const bool * labelMask)
{

	Volume_ComponentIndices ci;

	size_t N = ccl.res.x * ccl.res.y * ccl.res.z;
	//size_t Nbitmask = (N + 8-1) / 8; -> would have to do atomic add, or thread shuffle/poll
	managed_device_vector<uint> mask(N);
	//ci.mask.resize(N);

	ci.origToIndex.resize(N);

	/*
	set zero to everything other than component, set component to 1
	*/
	//component: 2
	//1,2,2,0,2,1,1
	//x,1,1,x,1,x,x -> mask
	{

		//Copy label selection mask to device
		bool * labelMaskDevice;
		size_t boolmapBytesize = 6 * numLabels * sizeof(bool);
		cudaMalloc(&labelMaskDevice, boolmapBytesize);
		cudaMemcpy(labelMaskDevice, labelMask, boolmapBytesize, cudaMemcpyHostToDevice);

		BLOCKS3D(8, ccl.res);
		__maskCCLKernel << < numBlocks, block >> > (
			ccl,			
			labelMaskDevice,
			THRUST_PTR(mask)			
			);

		cudaFree(labelMaskDevice);
	}
	

	uint NNZ = _scanAndCount<uint>(mask, ci.origToIndex);


	ci.indexToOrig.resize(NNZ);

	{
		BLOCKS3D(8, ccl.res);
		__indexToOrigKernel << < numBlocks, block >> > (
			ccl.res,			
			THRUST_PTR(mask),
			THRUST_PTR(ci.origToIndex),
			THRUST_PTR(ci.indexToOrig)
			);
	}

	//printf("N: %u, NNZ: %u, fract: %f\n", N, NNZ, float(NNZ) / N);


	/*thrust::host_vector<uint> mask_h = ci.mask;
	thrust::host_vector<uint> indices_h = tmpIndices;
	thrust::host_vector<uint> indexToOrig_h = ci.indexToOrig;

	uint * mask_h_ptr = mask_h.data();
	uint * indices_h_ptr = indices_h.data();
	uint * indexToOrig_h_ptr = indexToOrig_h.data();*/

	ci.NNZ = NNZ;
	ci.res = ccl.res;
	return ci;


	/*
		prefix sum 
	*/
	//0,0,1,1,2,2,2 -> indices
	//last element holds the count

	/*
		
	*/



	//stream compaction first

}

