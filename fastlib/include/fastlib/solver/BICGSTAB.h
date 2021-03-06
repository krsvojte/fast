#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/solver/Solver.h>


template <typename T>
struct LinearSys;

namespace fast {


	template <typename T>
	class BICGSTAB : public Solver<T> {	

	public:
		FAST_EXPORT BICGSTAB(bool verbose = false);
				

		FAST_EXPORT virtual typename Solver<T>::Output solve(
			LinearSystem * linearSys,
			const typename Solver<T>::SolveParams & solveParams
		) override;

		FAST_EXPORT virtual size_t requiredMemory(size_t NNZ) const override;
		
		
	};

	
	
	

}
