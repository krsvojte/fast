#pragma once
#ifdef _deprecated

#include <fastlib/FastLibDef.h>
#include <fastlib/volume/Volume.h>


#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>

namespace fast {

	template <typename T>
	class DiffusionSolver {


	public:

		using value_type = T;

		FAST_EXPORT DiffusionSolver(bool verbose = true);
		FAST_EXPORT ~DiffusionSolver();


		/*
			Assumes volChannel is synchronized on cpu
		*/
		FAST_EXPORT bool prepare(
			const Volume & volChannel,
			Dir dir,
			T d0,
			T d1,
			bool preserveAspectRatio = true
		);	

		//Returns current error
		FAST_EXPORT T solve(
			T tolerance,
			size_t maxIterations,
			size_t iterPerStep = size_t(-1)
		);

		//Returns false there's format mismatch and no conversion available
		FAST_EXPORT bool resultToVolume(Volume & vol);

		//Solves stable=state diffusion equation		
		//if dir is pos, 0 is high concetration, otherwise dim[dir]-1 is high
		FAST_EXPORT bool solve(
			Volume & volChannel, 						
			Volume * outVolume,
			Dir dir,
			float d0,
			float d1,
			float tolerance = 1.0e-6f			
		);

		FAST_EXPORT bool solveWithoutParticles(
			Volume & volChannel,
			Volume * outVolume,
			float d0,
			float d1,
			float tolerance = 1.0e-6f
		);

		FAST_EXPORT T tortuosity(
			const Volume & mask,			
			Dir dir
		);

		FAST_EXPORT T porosity() const { return _porosity; }

		FAST_EXPORT uint iterations() const { return _iterations;  }

	private:
			
		
		bool _verbose;

		Eigen::Matrix<T, Eigen::Dynamic, 1> _rhs;
		Eigen::Matrix<T, Eigen::Dynamic, 1> _x;
		Eigen::SparseMatrix<T, Eigen::RowMajor> _A;

		Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>
			/*,Eigen::IncompleteLUT<T>*/
		> _solver;


		T _porosity;

		uint _iterations;

	};



   
	
}
#endif