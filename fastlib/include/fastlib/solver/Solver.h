#pragma once


#include <fastlib/utility/Types.h>

template <typename T>
struct LinearSys;

template <typename T>
struct LinearSys_NU;



namespace fast {

	class Volume;
	class LinearSystem;
		

	template <typename T>
	class Solver {


	public:

		Solver(bool verbose) : _verbose(verbose), _iterations(0){

		}
		
		//deprecate
		struct PrepareParams {
			const Volume * mask = nullptr;			
			Volume * output = nullptr;
			Dir dir = X_NEG;
			T d0 = 0.0;
			T d1 = 1.0;
			vec3 cellDim = vec3(1.0,1.0,1.0);					
		};

		struct SolveParams {
			T tolerance;
			size_t maxIter;	
			int cpuThreads = -1;
		};

		enum Status {
			SOLVER_STATUS_SUCCESS,
			SOLVER_STATUS_DIVERGED,
			SOLVER_STATUS_NO_CONVERGENCE
		};

		struct Output {
			T error = 0;
			Status status = SOLVER_STATUS_NO_CONVERGENCE;
			size_t iterations = 0;
		};

		
		virtual Output solve(LinearSys<T> & system, const SolveParams & solveParams) { return Output(); };
		virtual Output solve(LinearSys_NU<T> & system, const SolveParams & solveParams) { return Output(); };
		
		virtual Output solve(const SolveParams & solveParams) { return Output(); };

		virtual Output solve(LinearSystem * system, const SolveParams & solveParams) { return Output(); }
		
		virtual size_t requiredMemory(size_t NNZ) const { return 0; };		


		bool isVerbose() const { return _verbose; }
	protected:
		bool _verbose;
		PrepareParams _params;
		size_t _iterations;
	};


}