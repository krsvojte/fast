#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/solver/Solver.h>


template <typename T>
struct LinearSys;

template <typename T>
struct LinearSys_NU;

namespace fast {


	template <typename T>
	class CG : public Solver<T> {

	public:
		FAST_EXPORT CG(bool verbose = false, bool precondition = true);

			
		FAST_EXPORT virtual typename Solver<T>::Output solve(
			LinearSystem * linearSys,
			const typename Solver<T>::SolveParams & solveParams
		) override;


		FAST_EXPORT virtual size_t requiredMemory(size_t NNZ) const override;

	private:
		bool _precondition;
	};

	



}
