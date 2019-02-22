#include "cuda/CudaUtility.h"


#include <iostream>
#include <vector>
#include <stdlib.h>

bool fast::cudaVerify(){

	std::cout << "Properties:" << std::endl;
	cudaPrintProperties();

	std::cout << "Memory info:" << std::endl;
	cudaPrintMemInfo();


	std::cout << "Malloc/copy test:" << std::endl;
	double * ptr;
	size_t N = 1024;
	_CUDA(cudaMalloc(&ptr,N * sizeof(double)));

	std::vector<double> arr(N);

	_CUDA(cudaMemcpy(ptr, arr.data(), N * sizeof(double), cudaMemcpyHostToDevice));

	_CUDA(cudaMemcpy(arr.data(),ptr, N * sizeof(double), cudaMemcpyDeviceToHost));

	_CUDA(cudaFree(ptr));

	return true;
	
}

bool fast::cudaCheck(
	cudaError_t result, 
	const char * function, 
	const char * file,
	int line,
	bool abort)
{

	if (result == cudaSuccess) 
		return true;
	
	std::cerr	<< "CUDA Error: " << cudaGetErrorString(result) 
				<< "(" << function << " at " 
				<< file << ":" << line
				<< ")" 
				<< std::endl;
	
	if (abort)
		exit(result);

	return false;
}

bool fast::cusolverCheck(cusolverStatus_t result, const char * function, const char * file, int line, bool abort /*= true*/)
{
	if (result == CUSOLVER_STATUS_SUCCESS)
		return true;

	std::cerr << "cuSolver Error: ";
	switch (result) {
		case CUSOLVER_STATUS_NOT_INITIALIZED:
			std::cerr << "CUSOLVER_STATUS_NOT_INITIALIZED, \
				\nThe cuSolver library was not initialized.This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSolver routine, or an error in the hardware setup. \
				\nTo correct : call cusolverCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.";
			break;
		case CUSOLVER_STATUS_ALLOC_FAILED:
			std::cerr << "CUSOLVER_STATUS_ALLOC_FAILED, \
				\nResource allocation failed inside the cuSolver library. This is usually caused by a cudaMalloc() failure.\
				\nTo correct : prior to the function call, deallocate previously allocated memory as much as possible.";
			break;
		case CUSOLVER_STATUS_INVALID_VALUE:
			std::cerr << "CUSOLVER_STATUS_INVALID_VALUE, \
				\nAn unsupported value or parameter was passed to the function (a negative vector size, for example).\
				\nTo correct : ensure that all the parameters being passed have valid values.";
			break;
		case CUSOLVER_STATUS_ARCH_MISMATCH:
			std::cerr << "CUSOLVER_STATUS_ARCH_MISMATCH, \
				\nThe function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision. \
				\nTo correct : compile and run the application on a device with compute capability 2.0 or above.";
			break;
		case CUSOLVER_STATUS_EXECUTION_FAILED:
			std::cerr << "CUSOLVER_STATUS_EXECUTION_FAILED, \
				\nThe GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\
				\nTo correct : check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.";
			break;
		case CUSOLVER_STATUS_INTERNAL_ERROR:
			std::cerr << "CUSOLVER_STATUS_INTERNAL_ERROR, \
				\nAn internal cuSolver operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\
				\nTo correct : check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine�s completion.";
			break;
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			std::cerr << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED, \
				\nThe matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\
				\nTo correct : check that the fields in descrA were set correctly.";
			break;
	}
	
	std::cerr	<< "(" << function << " at "
		<< file << ":" << line
		<< ")"
		<< std::endl;

	if (abort)
		exit(result);

	return false;
}

bool fast::cusparseCheck(cusparseStatus_t result, const char * function, const char * file, int line, bool abort /*= true*/)
{
	if (result == CUSPARSE_STATUS_SUCCESS)
		return true;

	std::cerr << "cuSparse Error: ";
	switch (result) {
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		std::cerr << "CUSPARSE_STATUS_NOT_INITIALIZED, \
				The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\
				To correct : call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";

		break;
	case CUSPARSE_STATUS_ALLOC_FAILED:
		std::cerr << "CUSPARSE_STATUS_ALLOC_FAILED, \
				Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\
				To correct : prior to the function call, deallocate previously allocated memory as much as possible.";
		break;
	case CUSPARSE_STATUS_INVALID_VALUE:
		std::cerr << "CUSPARSE_STATUS_INVALID_VALUE, \
				An unsupported value or parameter was passed to the function (a negative vector size, for example).\
				To correct : ensure that all the parameters being passed have valid values.";
		break;
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		std::cerr << "CUSPARSE_STATUS_ARCH_MISMATCH, \
				The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\
				To correct : compile and run the application on a device with appropriate compute capability, which is 1.1 for 32 - bit atomic operations and 1.3 for double precision.";
		break;
	case CUSPARSE_STATUS_MAPPING_ERROR:
		std::cerr << "CUSPARSE_STATUS_MAPPING_ERROR, \
				An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\
				To correct : prior to the function call, unbind any previously bound textures.";
		break;
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		std::cerr << "CUSPARSE_STATUS_EXECUTION_FAILED, \
				The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\
			To correct : check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
		break;
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		std::cerr << "CUSPARSE_STATUS_INTERNAL_ERROR, \
				An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\
				To correct : check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine�s completion.";
		break;
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		std::cerr << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, \
				The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\
				To correct : check that the fields in cusparseMatDescr_t descrA were set correctly.";
		break;
	}

	std::cerr << "(" << function << " at "
		<< file << ":" << line
		<< ")"
		<< std::endl;

	if (abort)
		exit(result);

	return false;
}

void fast::cudaPrintProperties()
{
	int dcount = 0;
	_CUDA(cudaGetDeviceCount(&dcount));

	float KB = 1024.f;
	float MB = 1024.0f * 1024.0f;

	for (auto i = 0; i < dcount; i++) {
		cudaDeviceProp p;
		_CUDA(cudaGetDeviceProperties(&p, i));

		std::cout << "Device ID " << i << std::endl;
		std::cout << "Name " << p.name << std::endl;
		std::cout << "---" << std::endl;

		std::cout << "Global Memory: " << p.totalGlobalMem / MB << "MB" << std::endl;
		std::cout << "Constant Memory: " << p.totalConstMem  / KB << "KB" << std::endl;
		std::cout << "Shared Memory / Block: " << p.sharedMemPerBlock / KB << "KB" << std::endl;
		std::cout << "Shared Memory / Multiprocessor: " << p.sharedMemPerMultiprocessor / KB << "KB" << std::endl;

		std::cout << "Registers / block: " << p.regsPerBlock << std::endl;

		std::cout << "---" << std::endl;
		std::cout << "Warp size: " << p.warpSize<< std::endl;
		std::cout << "Max threads per block: " << p.maxThreadsPerBlock << std::endl;
		std::cout << "Max thread dim: " << p.maxThreadsDim[0] << "x" << p.maxThreadsDim[1] << "x"  << p.maxThreadsDim[2]  << std::endl;
		std::cout << "Max grid size: " << p.maxGridSize[0] << "x" << p.maxThreadsDim[1] << "x" << p.maxThreadsDim[2] << std::endl;
		std::cout << "Multiprocessor count: " << p.multiProcessorCount << std::endl;
		std::cout << "Max threads per multiprocessor: " << p.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "Concurrent kernels: " << p.concurrentKernels << std::endl;
		std::cout << "Single to Double perf ratio: " << p.singleToDoublePrecisionPerfRatio<< std::endl;

		std::cout << "---" << std::endl;
		std::cout << "Kernel timout enabled: " << p.kernelExecTimeoutEnabled<< std::endl;
		std::cout << "---" << std::endl;


	}


}

void fast::cudaOccupiedMemory(size_t * total, size_t * occupied, int device)
{	
	size_t free;
	_CUDA(cudaMemGetInfo(&free, total));
	*occupied = *total - free;

}

void fast::cudaPrintMemInfo(int device /*= 0*/)
{
	size_t total, occupied;
	cudaOccupiedMemory(&total, &occupied);
	std::cout << occupied / (1024.0f * 1024.0f) << "MB / " << total / (1024.0f * 1024.0f) << "MB"
		<< " (" << (occupied / float(total)) * 100.0f << "%) \n";
}

