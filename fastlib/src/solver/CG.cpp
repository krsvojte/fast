#include "solver/CG.h"

#include "utility/DataPtr.h"
#include "volume/Volume.h"

#include "cuda/Volume.cuh"
#include "cuda/LinearSys.cuh"
#include "cuda/CudaUtility.h"
#include "cuda/CG.cuh"

#include "solver/LinearSystemDevice.h"
#include "solver/LinearSystemHost.h"

#include <Eigen/IterativeLinearSolvers>

#include "utility/Timer.h"

#include <iostream>
#include <fstream>


namespace fast {

	template <typename T>
	FAST_EXPORT CG<T>::CG(bool verbose, bool precondition) : Solver<T>(verbose), _precondition(precondition)
	{

	}



	template <typename T>
	FAST_EXPORT	size_t fast::CG<T>::requiredMemory(size_t nnz) const
	{
		return CGState<T>::requiredBytesize(nnz, _precondition);
	}


	template <typename T>
	FAST_EXPORT typename Solver<T>::Output fast::CG<T>::solve(LinearSystem * linearSys, const typename Solver<T>::SolveParams & solveParams)
	{

		
		assert(linearSys->type() == primitiveTypeof<T>());

		typename Solver<T>::Output out;
		LinearSystemDevice<T> * devicePtr = dynamic_cast<LinearSystemDevice<T>*>(linearSys);
		if (devicePtr) {
		
			//Try to allocate solver state
			CGState<T> state;
			try {
				state.resize(devicePtr->NNZ, _precondition);
			}
			catch (thrust::system_error & er) {
				std::cerr << er.what() << std::endl;
				return out;
			}

			bool res = CG_Solve<T>(
				*devicePtr,
				state,
				solveParams.tolerance,
				solveParams.maxIter,
				&out.error,
				&out.iterations,
				Solver<T>::isVerbose()
				);

			out.status = res ? Solver<T>::Status::SOLVER_STATUS_SUCCESS : Solver<T>::Status::SOLVER_STATUS_NO_CONVERGENCE;
			return out;			
		}


		auto hostPtr = dynamic_cast<LinearSystemHost<T>*>(linearSys);
		if (hostPtr) {

			int threadNum = omp_get_num_procs();
			if (solveParams.cpuThreads > 0) {
				threadNum = solveParams.cpuThreads;
			}
			omp_set_num_threads(threadNum);
			Eigen::setNbThreads(threadNum);

			Eigen::ConjugateGradient<Eigen::SparseMatrix<T, Eigen::RowMajor>, Eigen::Lower | Eigen::Upper> 
				_solver;

			_solver.compute(hostPtr->A);
			_solver.setTolerance(solveParams.tolerance);
			_solver.setMaxIterations(solveParams.maxIter);

			hostPtr->x = _solver.solveWithGuess(hostPtr->b, hostPtr->x);

			out.error = _solver.error();
			if (_solver.info() == Eigen::ComputationInfo::Success)
				out.status = Solver<T>::Status::SOLVER_STATUS_SUCCESS;
			else
				out.status = Solver<T>::Status::SOLVER_STATUS_NO_CONVERGENCE;

			out.iterations = _solver.iterations();
			//copyHostToDevice(_x.data(), sys.x);

			return out;
		}

		//Linear system derivation not implemented
		assert(false);
		return out;
	}



	template class CG<double>;
	template class CG<float>;
}
