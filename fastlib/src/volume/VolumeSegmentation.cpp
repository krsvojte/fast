#include "volume/VolumeSegmentation.h"

#include "volume/Volume.h"
#include "cuda/VolumeCCL.cuh"

namespace fast {


	const bool * VolumeCCL::getDirMask(Dir dir) const
	{
		return reinterpret_cast<const bool*>(boundaryLabelMask.data()) + (numLabels * uint(dir));			
	}	
	
	std::vector<uchar> VolumeCCL::getDirMask(Dir begin, Dir end) const
	{
		std::vector<uchar> mask(numLabels,0);

		const bool * beginMask = getDirMask(begin);
		const bool * endMask = getDirMask(end);
		for (int i = 0; i < numLabels; i++) {
			mask[i] = beginMask[i] && endMask[i];
		}

		return mask;	}

	FAST_EXPORT VolumeCCL getVolumeCCL(const Volume & mask, uchar background)
	{
		return getVolumeCCL(*mask.getCUDAVolume(), background);
		
	}


	FAST_EXPORT VolumeCCL getVolumeCCL(const CUDA_Volume & mask, uchar background)
	{
		VolumeCCL ccl;

		ccl.labels = std::make_shared<Volume>(
				ivec3( mask.res.x, mask.res.y, mask.res.z), TYPE_UINT
			);


		ccl.background = background;

		ccl.numLabels = VolumeCCL_Label(
			mask,
			*ccl.labels->getCUDAVolume(),
			background
		);


		ccl.boundaryLabelMask.clear();
		ccl.boundaryLabelMask.resize(ccl.numLabels * 7);



		VolumeCCL_BoundaryLabels(
			*ccl.labels->getCUDAVolume(),
			ccl.numLabels,
			(bool*)ccl.boundaryLabelMask.data()
		);


		return ccl;
	}

	FAST_EXPORT Volume generateBoundaryConnectedVolume(const VolumeCCL & ccl, Dir dir, bool invertMask)
	{
		
		Volume boundary(ccl.labels->dim(), TYPE_UCHAR);

		

		std::vector<uchar> mask(ccl.numLabels, 0);
		memcpy(mask.data(), ccl.getDirMask(dir), ccl.numLabels * sizeof(bool));
		if (invertMask) {
			for (auto i = 1; i < mask.size(); i++) {				
				mask[i] = !mask[i];
			}				
		}
		

		VolumeCCL_GenerateVolume(
			*ccl.labels->getCUDAVolume(),
			ccl.numLabels,
			reinterpret_cast<const bool*>(mask.data()),
			*boundary.getCUDAVolume()
		);

		return boundary;
	}

	FAST_EXPORT Volume generateCCLVisualization(
		const VolumeCCL & ccl,
		Volume * mask
	)
	{
		Volume outViz(ccl.labels->dim(), TYPE_UCHAR4);
		if (mask) {
			VolumeCCL_Colorize(*ccl.labels->getCUDAVolume(), *outViz.getCUDAVolume(), mask->getCUDAVolume(), 
				(ccl.background) ? 0 : 255
				);
		}
		else {
			VolumeCCL_Colorize(*ccl.labels->getCUDAVolume(), *outViz.getCUDAVolume());
		}
		return outViz;		
	}

	

	FAST_EXPORT std::vector<std::unique_ptr<fast::Volume>> generateSeparatedCCLVolumes(const VolumeCCL & ccl)
	{
		std::vector<std::unique_ptr<fast::Volume>> result;
		for (auto i = 0; i < ccl.numLabels; i++) {
			
			auto vol = std::make_unique<Volume>(ccl.labels->dim(), TYPE_UCHAR);
			
			std::vector<uchar> mask(ccl.numLabels,0);
			mask[i] = 255;


			VolumeCCL_GenerateVolume(
				*ccl.labels->getCUDAVolume(),
				ccl.numLabels,
				reinterpret_cast<const bool*>(mask.data()),
				*vol->getCUDAVolume()
			);

			result.push_back(std::move(vol));
		}

		return result;
	}

	FAST_EXPORT std::vector<fast::CCLSegmentInfo> getCCLSegmentInfo(
		const VolumeCCL & ccl		
	)
	{
		
		assert(ccl.labels->type() == TYPE_UINT);
		uint * labels = (uint*)(ccl.labels->getPtr().getCPU());

		


		auto dim = ccl.labels->dim();
		std::vector<fast::CCLSegmentInfo> resultAll(ccl.numLabels);
		int cnt = 0;
		for (auto & r : resultAll) {
			r.minBB = dim;
			r.maxBB = ivec3(0);
			r.voxelNum = 0;
			r.labelID = cnt++;
		}

		
		for (auto z = 0; z < dim.z; z++) {
			for (auto y = 0; y < dim.y; y++) {
				for (auto x = 0; x < dim.x; x++) {
					ivec3 ipos = ivec3(x, y, z);
					auto i = linearIndex(dim, ipos);					
					auto &r = resultAll[labels[i]];
					r.voxelNum++;
					r.maxBB = glm::max(r.maxBB, ipos);
					r.minBB = glm::min(r.minBB, ipos);
				}
			}
		}

		for (auto & r : resultAll) {
			r.maxBB += ivec3(1); //Non inclusive bounds
			r.atBoundary =	(r.maxBB.x == dim.x || r.maxBB.y == dim.y || r.maxBB.z == dim.z) ||
							(r.minBB.x == 0 || r.minBB.y == 0 || r.minBB.z == 0);
		}

		return resultAll;

		

	}

	FAST_EXPORT std::vector<fast::CCLSegmentInfo> filterSegmentInfo(
		const std::vector<CCLSegmentInfo> & sinfo,
		const VolumeCCL & ccl, 
		Dir dir /*= DIR_NONE*/, 
		bool invertMask /*= true*/)
	{

		//Construct mask
		std::vector<uchar> mask(ccl.numLabels, 0);
		memcpy(mask.data(), ccl.getDirMask(dir), ccl.numLabels * sizeof(bool));
		if (invertMask) {
			for (auto i = 1; i < mask.size(); i++)
				mask[i] = !mask[i];
		}

		//Filter result
		std::vector<fast::CCLSegmentInfo> result;
		for (auto i = 0; i < sinfo.size(); i++) {
			if (mask[i])
				result.push_back(sinfo[i]);
		}

		return result;
	}

	FAST_EXPORT float getCCLSegmentRatio(const std::vector<CCLSegmentInfo> & sinfo, const VolumeCCL & ccl)
	{
		int vox = 0;
		for (auto & inf : sinfo) {
			vox += inf.voxelNum;
		}
		auto dim = ccl.labels->dim();
		return float(vox) / (dim.x*dim.y*dim.z);
	}

}

