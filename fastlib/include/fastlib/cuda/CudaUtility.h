#pragma once


#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

#include <fastlib/FastLibDef.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define _CUDA(x) fast::cudaCheck(x, TOSTRING(x), __FILE__, __LINE__)
#define _CUSOLVER(x) fast::cusolverCheck(x, TOSTRING(x), __FILE__, __LINE__)
#define _CUSPARSE(x) fast::cusparseCheck(x, TOSTRING(x), __FILE__, __LINE__)

namespace fast {

	FAST_EXPORT bool cudaCheck(
		cudaError_t result,
		const char * function,
		const char * file,
		int line,
		bool abort = true);

	FAST_EXPORT bool cusolverCheck(
		cusolverStatus_t result,
		const char * function,
		const char * file,
		int line,
		bool abort = true);

	FAST_EXPORT bool cusparseCheck(
		cusparseStatus_t result,
		const char * function,
		const char * file,
		int line,
		bool abort = true);

	FAST_EXPORT void cudaPrintProperties();

	FAST_EXPORT void cudaOccupiedMemory(size_t * total, size_t * occupied, int device = 0);

	FAST_EXPORT void cudaPrintMemInfo(int device = 0);

	FAST_EXPORT bool cudaVerify();


	class CUDATimer {		
	public:

		CUDATimer(bool autoStart = false) {
			_running = false;

			cudaEventCreate(&_startEvent);
			cudaEventCreate(&_stopEvent);
			if (autoStart) start();
		}

		~CUDATimer() {
			cudaEventDestroy(_startEvent);
			cudaEventDestroy(_stopEvent);
		}

		void start() {			
			cudaEventRecord(_startEvent);
			_running = true;
		}

		void stop() {
			_running = false;
			cudaEventRecord(_stopEvent);
		}
	
		float timeMs() {
			if (_running) stop();
			cudaEventSynchronize(_stopEvent);
			float ms = 0;
			cudaEventElapsedTime(&ms, _startEvent, _stopEvent);
			return ms;
		}

		//Time in seconds
		float time() {
			return timeMs() / 1000.0f;
		}

		

	private:
		bool _running;
		cudaEvent_t _startEvent;
		cudaEvent_t _stopEvent;
	};

	


#define CUDA_TIMER(expr,name,prof)		\
	{CUDATimer _localTimer(true); expr; prof.add(name, _localTimer.time());} \


}
