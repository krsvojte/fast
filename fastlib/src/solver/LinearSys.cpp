/*
#include "solver/LinearSys.h"

#include "volume/Volume.h"
#include "cuda/Volume.cuh"
#include "cuda/CudaUtility.h"
#include "cuda/LinearSys.cuh"

#include "volume/VolumeSegmentation.h"
#include "cuda/VolumeCCL.cuh"

namespace fast {

	LinearSys_SysParams setupSysParams(const LinearSys_PrepareParams & params) {
		LinearSys_SysParams sysp;
		sysp.highConc = double(1.0);
		sysp.lowConc = double(0.0);
		sysp.concetrationBegin = (getDirSgn(params.dir) == 1) ? sysp.highConc : sysp.lowConc;
		sysp.concetrationEnd = (getDirSgn(params.dir) == 1) ? sysp.lowConc : sysp.highConc;

		sysp.cellDim[0] = params.cellDim.x;
		sysp.cellDim[1] = params.cellDim.y;
		sysp.cellDim[2] = params.cellDim.z;
		sysp.faceArea[0] = sysp.cellDim[1] * sysp.cellDim[2];
		sysp.faceArea[1] = sysp.cellDim[0] * sysp.cellDim[2];
		sysp.faceArea[2] = sysp.cellDim[0] * sysp.cellDim[1];

		sysp.dirPrimary = getDirIndex(params.dir);
		sysp.dirSecondary = make_uint2((sysp.dirPrimary + 1) % 3, (sysp.dirPrimary + 2) % 3);

		sysp.dir = params.dir;
		return sysp;
	}

	template <typename T>
	FAST_EXPORT std::unique_ptr<LinearSys<T>>
		GenerateLinearSystem(const LinearSys_PrepareParams & params)
	{

		std::unique_ptr<LinearSys<T>> sysptr;

		const auto dim = params.mask->res;
		const size_t N = dim.x * dim.y * dim.z;

		/ *
		Generate continous domain
		* /
		//std::shared_ptr<CUDA_Volume> domain;
		//Volume v;
		
		Volume v({ dim.x,dim.y,dim.z }, primitiveTypeof<T>());
		CUDA_Volume domain = *v.getCUDAVolume();
		//domain = v.getCUDAVolume(v.addChannel({ dim.x,dim.y,dim.z }, primitiveTypeof<T>(), false, "domain"));
		{			
			fast::CUDATimer tGenDomain(true);

			LinearSys_GenerateDomain(*params.mask, params.d0, params.d1, domain);

			if(params.verbose)
				std::cout << "Gen domain " << tGenDomain.time() << "s" << std::endl;
		}


		
		if (!LinearSys_commitSysParams(setupSysParams(params))) {
			return nullptr;
		}

		if (params.verbose)
			std::cout << "Commit sys params" << std::endl;


		/ *
		Generate linear sys
		* /
		CUDATimer tsysgen(true);
		sysptr = std::move(LinearSys_GenerateSystem<T>(domain));
		tsysgen.stop();

		//Delete volume with domain -> no longer needed
		v.clear();
		
		if (params.verbose) {
			cudaPrintMemInfo();
			std::cout << "LinearSys_GenerateSystem: " << tsysgen.timeMs() << "ms" << std::endl;
		}


		return sysptr;
		

	}

	template <typename T>
	FAST_EXPORT std::unique_ptr<LinearSys_NU<T>>
		GenerateLinearSystem_NonUniform(const LinearSys_PrepareParams & params)
	{

		std::unique_ptr<LinearSys_NU<T>> sysptr;

		const auto dim = params.mask->res;
		const size_t N = dim.x * dim.y * dim.z;

		CUDATimer tccl(true);

		//Get connected components
		//uint for every voxel
		auto ccl = std::make_unique<VolumeCCL>(getVolumeCCL(*params.mask, 255));
		tccl.stop();
		if (params.verbose) {
			cudaPrintMemInfo();
			std::cout << "getVolumeCCL: " << tccl.timeMs() << "ms" << std::endl;
		}

		const Dir dir0 = params.dir;
		const Dir dir1 = getOppositeDir(params.dir);


		CUDATimer tcclLin(true);
		std::unique_ptr<Volume_ComponentIndices> indices = std::make_unique<Volume_ComponentIndices>(
			std::move(
				VolumeCCL_LinearizeComponents(*ccl->labels->getCUDAVolume(), ccl->numLabels, (bool*)ccl->getDirMask(dir0, dir1).data())
			)
			);

		tcclLin.stop();

		//Free CCL
		ccl = nullptr;

		if (params.verbose) {
			cudaPrintMemInfo();
			std::cout << "_LinearizeComponent: " << tcclLin.timeMs() << "ms" << std::endl;
		}

		if (!LinearSys_commitSysParams(setupSysParams(params))) {
			return nullptr;
		}

		if (params.verbose)
			std::cout << "Commit sys params" << std::endl;


		CUDATimer taxb(true);
		sysptr = std::move(LinearSys_GenerateSystem_NU<T>(*indices));
		taxb.stop();

		sysptr->indices = std::move(indices);
		
		if (params.verbose) {
			cudaPrintMemInfo();
			std::cout << "LinearSys_GenerateSystem_NU: " << taxb.timeMs() << "ms" << std::endl;
		}


		return sysptr;

	}




	template <typename T>
	FAST_EXPORT std::vector<T> GetLinearSysSolutionSlice(const LinearSys<T> & sys, Dir dir)
	{
		std::vector<T> slice;
		int k = getDirIndex(dir);
		ivec3 dim = { sys.res.x, sys.res.y, sys.res.z };
		size_t N = dim[(k + 1) % 3] * dim[(k + 2) % 3];
		slice.resize(N);
		copySlice<T>(sys.x, sys.res,  dir, slice.data(), nullptr);
		return slice;
	}

	template <typename T>
	FAST_EXPORT std::vector<T> GetLinearSysSolutionSlice(const LinearSys_NU<T> & sys, Dir dir)
	{
		std::vector<T> slice;
		int k = getDirIndex(dir);
		ivec3 dim = { sys.res.x, sys.res.y, sys.res.z };
		size_t N = dim[(k + 1) % 3] * dim[(k + 2) % 3];
		slice.resize(N);
		copySlice<T>(sys.x, sys.res, dir, slice.data(), sys.indices.get());
		return slice;
	}





	template FAST_EXPORT std::vector<double> GetLinearSysSolutionSlice(const LinearSys<double> & sys, Dir dir);
	template FAST_EXPORT std::vector<float> GetLinearSysSolutionSlice(const LinearSys<float> & sys, Dir dir);
	template FAST_EXPORT std::vector<double> GetLinearSysSolutionSlice(const LinearSys_NU<double> & sys, Dir dir);
	template FAST_EXPORT std::vector<float> GetLinearSysSolutionSlice(const LinearSys_NU<float> & sys, Dir dir);

	

	template FAST_EXPORT std::unique_ptr<LinearSys<double>> GenerateLinearSystem(const LinearSys_PrepareParams & params);
	template FAST_EXPORT std::unique_ptr<LinearSys<float>> GenerateLinearSystem(const LinearSys_PrepareParams & params);

	template FAST_EXPORT std::unique_ptr<LinearSys_NU<double>> GenerateLinearSystem_NonUniform(const LinearSys_PrepareParams & params);
	template FAST_EXPORT std::unique_ptr<LinearSys_NU<float>> GenerateLinearSystem_NonUniform(const LinearSys_PrepareParams & params);

}*/