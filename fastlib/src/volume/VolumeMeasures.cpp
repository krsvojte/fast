#include "volume/VolumeMeasures.h"

#include "volume/Volume.h"

#include "solver/BICGSTAB.h"
#include "solver/CG.h"
#include "solver/Diffusion.h"
#include "solver/LinearSystemDevice.h"
#include "solver/LinearSystemHost.h"

#include "solver/DiffusionSolver.h"
#include "volume/VolumeSegmentation.h"

#include "cuda/VolumeSurface.cuh"
#include "volume/VolumeSurface.h"


#include "solver/LinearSys.h"
#include "cuda/LinearSys.cuh"



#include "cuda/CudaUtility.h"

#include <numeric>
#include <iostream>
#include <vector>

#include <glm/gtc/constants.hpp>

namespace fast {
	
	
	size_t freeMemory() {
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		return free;
	}


	template <typename T>
	FAST_EXPORT T getTortuosity(
		const Volume & mask, 
		const TortuosityParams & params, 
		DiffusionSolverType solverType, 
		Volume * concetrationOutput,
		size_t * iterationsOut
	)
	{
		if (mask.type() != TYPE_UCHAR) {
			std::cerr << "Mask must by type uchar" << std::endl;
			return T(0);
		}

		if (mask.getPtr().getCPU() == nullptr) {
			std::cerr << "Mask must be allocated and retrieved to CPU" << std::endl;
			return T(0);
		}	
				
		T tau = 0.0;
		auto maxDim = std::max(mask.dim().x, std::max(mask.dim().y, mask.dim().z));
		auto minDim = std::min(mask.dim().x, std::min(mask.dim().y, mask.dim().z));	

		DiffusionPrepareParams dp;
		dp.dir = params.dir;
		dp.d0 = double(params.coeffs[0]);
		dp.d1 = double(params.coeffs[1]);
		dp.cellDim = vec3(1.0f / maxDim);
		dp.type = primitiveTypeof<T>();
		

		std::unique_ptr<LinearSystem> linearSystem;
		
		if(params.onDevice)
			linearSystem = GenerateDiffusionLinearSystemDevice(dp, mask.getCUDAVolume());			
		else
			linearSystem = GenerateDiffusionLinearSystemHost(
				dp, 
				mask.dim(), 
				static_cast<const uchar*>(mask.getPtr().getCPU()),
				params.cpuThreads
			);
		

		assert(linearSystem);
		
		/*
		// Instantiate solver
		*/
		std::unique_ptr<Solver<T>> solver;
		switch (solverType) {
		case DSOLVER_BICGSTAB:		
			solver = std::make_unique<BICGSTAB<T>>(params.verbose);	
			break;
		case DSOLVER_CG:
			solver = std::make_unique<CG<T>>(params.verbose);
			break;		
		default:
			std::cerr << "Unknown solver" << std::endl;
			return 0;		
		}
			

		size_t freeMem = freeMemory() - 64 * 1024; //-64kb, buffer for temporary objects
		size_t reqMem = solver->requiredMemory(linearSystem->NNZ);
		float gb = 1024.0f * 1024 * 1024;
		if (freeMem < reqMem) {
			std::cerr << "Warning: Solver will not fit in GPU memory." << std::endl;
			std::cerr << "Warning: Free " << freeMem / gb << "GB" << std::endl;
			std::cerr << "Warning: Required " << reqMem / gb << "GB" << std::endl;

			ivec3 suggestDim = mask.dim();
			while (solver->requiredMemory(suggestDim.x*suggestDim.y*suggestDim.z) < freeMem && freeMem > 0) {
				suggestDim -= vec3(1);
			}
			std::cerr << "Warning: Reduce resolution to: " << suggestDim.x << "x" << suggestDim.y << "x" << suggestDim.z << std::endl;
			std::cerr << "Warning: Or use single precision " << std::endl;

			if (params.oversubscribeGPUMemory) {
				std::cerr << "Oversubscription is enabled (Must have Pascal or newer and running on Unix/Mac)" << std::endl;
			}
			else {
				return T(0);
			}
		}
		if (params.verbose) {
			std::cout << "Solver will need " << reqMem / gb << "GB out of " << freeMem / gb << " GB available memory" << std::endl;
		}
		
		typename Solver<T>::SolveParams sp;
		sp.maxIter = params.maxIter;
		sp.tolerance = params.tolerance;
		sp.cpuThreads = params.cpuThreads;

		typename Solver<T>::Output out;

		out = solver->solve(linearSystem.get(), sp);

		if (iterationsOut) 
			*iterationsOut = out.iterations;

		
		/*
		Optional output
		*/
		if (concetrationOutput && params.onDevice) {			
			auto * linSysDevice = static_cast<LinearSystemDevice<T>*>(linearSystem.get());
			copyVectorToVolume<T>(linSysDevice->x, *concetrationOutput->getCUDAVolume());
		}
		else {
			auto * linSysDevice = static_cast<LinearSystemHost<T>*>(linearSystem.get());
#ifdef _DEBUG
			std::cerr << "Output not supported yet VolumeMeasures" << std::endl;
#endif
		}
	
		if (out.status != Solver<T>::SOLVER_STATUS_SUCCESS) {
			return T(0);
		}	

		
		std::vector<T> outputVolume;
		T * volumeHost = nullptr;

		if (params.onDevice) {
			auto * linSysDevice = static_cast<LinearSystemDevice<T>*>(linearSystem.get());
			outputVolume.resize(linSysDevice->NNZ);
			copyDeviceToHost(linSysDevice->x, outputVolume.data());
			volumeHost = outputVolume.data();
		}		
		else {
			auto * linSysDevice = static_cast<LinearSystemHost<T>*>(linearSystem.get());
			volumeHost = linSysDevice->x.data();
		}

		/*
			Get porosity
		*/
		const T porosity = params.porosityPrecomputed ?
			static_cast<T>(params.porosity) :
			getPorosity<T>(mask);


		/*
			Tortuosity
		*/

		{	
			const uchar * maskData = static_cast<const uchar*>(mask.getPtr().getCPU());
			const ivec3 dim = mask.dim();
			const int primaryDim = getDirIndex(params.dir);
			const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };		

			//Number of elems in plane
			const int n = dim[secondaryDims[0]] * dim[secondaryDims[1]];

			ivec3 dpos = ivec3(0);
			dpos[primaryDim] = getDirSgn(params.dir);

			//Index in primary dim
			const int zeroPlane = (getDirSgn(params.dir) == -1) ? 0 : dim[primaryDim] - 1;		

				

			T dcSumOverPlanes = 0;			
			T dc0;

			for (int k = 0; k < dim[primaryDim]; k++) {
				T sumDConcetration = T(0);
				size_t poreCount = 0;

				for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
					for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
						ivec3 pos;
						pos[primaryDim] = k;
						pos[secondaryDims[0]] = i;
						pos[secondaryDims[1]] = j;

						//size_t sliceIndex = linearIndex(ivec3(dim[secondaryDims[0]], dim[secondaryDims[1]], 1), { i,j,0 });
						//size_t maskIndex = linearIndex(dim, pos);
						size_t volIndex = linearIndex(dim, pos);
						
					
						if (maskData[volIndex] == 0) {
							poreCount++;
						}

						if(maskData[volIndex] != 0 && params.useNonUniform){
							continue;
						}

						T coeffHere = params.coeffs[(maskData[volIndex] == 0) ? 0 : 1];
						T concHere = volumeHost[volIndex];
						T concPrev = T(0.0);
						T concDx = T(0.5);


						if (isValidPos(dim, pos + dpos)) {
							size_t volIndexPrev = linearIndex(dim, pos + dpos);
							concPrev = volumeHost[volIndexPrev];// *params.coeffs[(maskData[volIndexPrev] == 0) ? 0 : 1];
							concDx = T(1.0);			
							if (params.useNonUniform) {
								if (maskData[volIndexPrev] != 0)
									concDx = T(0.5);
							}
						}						

						/*if ((concHere - concPrev) < 0) {
							char b;
							b = 0;
						}*/
						sumDConcetration += coeffHere * (concHere - concPrev) / concDx;
						//T concHere = slice[sliceIndex];
						
						//int coeffIndex = ;
						//sumConcetration += concHere * params.coeffs[(maskData[volIndex] == 0) ? 0 : 1];
					}
				}			
				
				
				T dcPlane = sumDConcetration / n;
				if (params.useNonUniform) {
					dcPlane = sumDConcetration / poreCount;
				}

				dcSumOverPlanes += dcPlane;

				if (k == zeroPlane) {
					dc0 = dcPlane;
				}
				
			}	

			T dc = dcSumOverPlanes / dim[primaryDim];
			tau = porosity / (dc * dim[primaryDim]);

			T tau0 = porosity / (dc0 * dim[primaryDim]);
			//std::cout << "tau0 = " << tau0 << " | tavg = " << tau << " -> " << ((tau0) / tau - 1.0) * 100 << "% diff" << std::endl;

		}


		return tau;
		
	}


	template <typename T>
	FAST_EXPORT T getPorosity(const Volume & mask)
	{
		return T(mask.sumZeroElems()) / T(mask.totalElems());		
	}

	template <typename T>
	FAST_EXPORT T getReactiveAreaDensity(const Volume & mask, ivec3 res, float isovalue)
	{
		Volume areas = getVolumeArea(mask, res, isovalue);
		double area = 0.0f;
		areas.sum(&area);
		return T(area);	
	}

	template <typename T>
	FAST_EXPORT std::array<T, 6> getReactiveAreaDensityTensor(const VolumeCCL & ccl, ivec3 res, float isovalue)
	{

		std::array<T, 6> result;

		VolumeSurface_MCParams params;
		if (res.x == 0 || res.y == 0 || res.z == 0) {
			params.res = make_uint3(ccl.labels->dim().x, ccl.labels->dim().y, ccl.labels->dim().z);
		}
		else {
			params.res = make_uint3(res.x, res.y, res.z);
		}
		params.isovalue = isovalue;
		params.smoothingOffset = 1.0f;

		
		for (auto i = 0; i < 6; i++) {
			
			//Generate volume of empty space that is connected to i'th boundary
			Volume boundaryVolume = fast::generateBoundaryConnectedVolume(ccl, Dir(i));
			boundaryVolume.getPtr().createTexture();

			//Run marching cubes on the volume
			Volume areas = getVolumeArea(boundaryVolume, res, isovalue);

			double area = 0.0f;
			areas.sum(&area);
			result[i] = T(area);
		}

		return result;
	}

	template <typename T>
	FAST_EXPORT T getShapeFactor(T averageParticleArea, T averageParticleVolume)
	{	
		const T numeratorCoeff = glm::pow(T(3.0 / (4.0 * glm::pi<T>())), T(1.0 / 3.0));

		return (numeratorCoeff * averageParticleArea) / glm::pow(averageParticleVolume, T(2.0 / 3.0));
	}

	
	

	template FAST_EXPORT double getTortuosity<double>(const Volume &, const TortuosityParams &, DiffusionSolverType, Volume *, size_t *);
	template FAST_EXPORT float getTortuosity<float>(const Volume &, const TortuosityParams &, DiffusionSolverType, Volume *, size_t *);



	template FAST_EXPORT float getPorosity<float>(const Volume &);
	template FAST_EXPORT double getPorosity<double>(const Volume &);

	template FAST_EXPORT float getReactiveAreaDensity<float>(const Volume &, ivec3, float);
	template FAST_EXPORT double getReactiveAreaDensity<double>(const Volume &, ivec3, float);

	template FAST_EXPORT std::array<double, 6> getReactiveAreaDensityTensor(const VolumeCCL & ccl, ivec3 res, float isovalue);
	template FAST_EXPORT std::array<float, 6> getReactiveAreaDensityTensor(const VolumeCCL & ccl, ivec3 res, float isovalue);


	template FAST_EXPORT double getShapeFactor<double>(double,double);
	template FAST_EXPORT float getShapeFactor<float>(float, float);
}
