#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/utility/Types.h>

#include <memory>
#include <vector>


struct CUDA_Volume;

namespace fast {
	
	class Volume;
	


	struct VolumeCCL {
		uint numLabels; //Includes 0 label -> background		
		std::shared_ptr<Volume> labels;
		const bool * getDirMask(Dir dir) const;

		/*
			Return mask of labels that connec both begin and end boundaries
		*/
		std::vector<uchar> getDirMask(Dir begin, Dir end) const;

		std::vector<uchar> boundaryLabelMask;
		uchar background;

	};

	FAST_EXPORT VolumeCCL getVolumeCCL(
		const Volume & mask,
		uchar background
	);

	FAST_EXPORT VolumeCCL getVolumeCCL(
		const CUDA_Volume & mask,
		uchar background
	);

	FAST_EXPORT Volume generateBoundaryConnectedVolume(
		const VolumeCCL & ccl,
		Dir dir,
		bool invertMask = false
	);

	FAST_EXPORT Volume generateCCLVisualization(
		const VolumeCCL & ccl,
		Volume * mask = nullptr
	);

	FAST_EXPORT std::vector<std::unique_ptr<Volume>> generateSeparatedCCLVolumes(
		const VolumeCCL & ccl
	);

	struct CCLSegmentInfo {
		ivec3 minBB;
		ivec3 maxBB;
		int voxelNum;
		bool atBoundary;
		int labelID;
	};

	/*
		Volume must be retrieved to CPU prior to calling this function
	*/
	FAST_EXPORT std::vector<CCLSegmentInfo> getCCLSegmentInfo(const VolumeCCL & ccl);

	FAST_EXPORT std::vector<CCLSegmentInfo> filterSegmentInfo(const std::vector<CCLSegmentInfo> & sinfo, const VolumeCCL & ccl, Dir dir = DIR_NONE, bool invertMask = true);

	FAST_EXPORT float getCCLSegmentRatio(const std::vector<CCLSegmentInfo> & sinfo, const VolumeCCL & ccl);


}