#include "BICGSTAB.cuh"

#include <iostream>
template <typename T>
bool BICGState<T>::resize(size_t N)
{
	r0.resize(N);
	r.resize(N);
	p.resize(N);
	v.resize(N);
	h.resize(N);
	s.resize(N);
	t.resize(N);
	y.resize(N);
	z.resize(N);
	return true;
}





template <typename T>
bool fast::BICG_Solve(LinearSystemDevice<T> & sys, BICGState<T> & state, T tolerance, size_t maxIter, T *outError, size_t * outIterations, bool verbose)
{




	size_t iterations = 0;


	const int perBlock2D = 32;
	const int perBlock3D = 8;

	T tol = tolerance;
	T tol_error = 1.0;

	size_t maxIters = maxIter;


	///***//
	//thrust::fill(sys.x.begin(), sys.x.end(), T(0));
	///***//

	//size_t n = _sys.res.x*_sys.res.y*_sys.res.z;

	//1. Residual r
	fast::LinearSys_Residual<T>(sys, state.r);




	//2. Choose rhat0 ..
	thrust::copy(state.r.begin(), state.r.end(), state.r0.begin());




	T r0_sqnorm = fast::squareNorm(state.r0);
	T rhs_sqnorm = fast::squareNorm(sys.b);




	if (rhs_sqnorm == 0) {
		thrust::fill(sys.x.begin(), sys.x.end(), T(0));
		return T(0);
	}

	T rho = 1;
	T alpha = 1;
	T w = 1;

	thrust::fill(state.v.begin(), state.v.end(), T(0));
	thrust::fill(state.p.begin(), state.p.end(), T(0));

	T tol2 = tol*tol*rhs_sqnorm;
	T eps2 = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon();

	size_t i = 0;
	size_t restarts = 0;

	size_t NNZ = sys.NNZ;

	T rsqNorm = 0;
	rsqNorm = fast::squareNorm<managed_device_vector<T>>(state.r);


	bool dv = false; //debug verbose


	while (rsqNorm > tol2 && i < maxIters)
	{
		T rho_old = rho;

		rho = fast::dotProduct<managed_device_vector<T>>(state.r0, state.r);

		if (dv) std::cout << "BICGSTAB: " << i << std::endl;
		if (dv) std::cout << "\t" << "sq r: " << rsqNorm << std::endl;
		if (dv) std::cout << "\t" << " rho: " << rho << std::endl;

		if (abs(rho) < eps2*r0_sqnorm)
		{
			fast::LinearSys_Residual<T>(sys, state.r);
			thrust::copy(state.r.begin(), state.r.end(), state.r0.begin());
			rho = r0_sqnorm = fast::squareNorm<managed_device_vector<T>>(state.r);
			restarts++;
			//printf("restart at %u iteration\n", i);
		}

		T beta = (rho / rho_old) * (alpha / w);
		if (dv) std::cout << "\t" << " beta: " << beta << std::endl;

		{
			//p = r + beta * (p - w * v);
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__aPlusBetaBGammaPlusCKernel << <grid, block >> > (NNZ, THRUST_PTR(state.p), THRUST_PTR(state.v), THRUST_PTR(state.r), -w, beta);
		}

		if (dv) std::cout << "\t" << " psq: " << fast::squareNorm(state.p) << std::endl;




		//Preconditiner a)
		{
			{
				BLOCKS_GRID2D(perBlock2D, NNZ);
				fast::__mulByInverseKernel << < grid, block >> > (NNZ, THRUST_PTR(sys.A.dir[DIR_NONE]), THRUST_PTR(state.p), THRUST_PTR(state.y));
			}
			if (dv) std::cout << "\t" << " ysq: " << fast::squareNorm(state.y) << std::endl;
			{
				BLOCKS3D(perBlock3D, sys.res);
				fast::__matrixVecProductKernel << < numBlocks, block >> > (sys.res, sys.A.getPtr(), THRUST_PTR(state.y), THRUST_PTR(state.v));
			}
			if (dv) std::cout << "\t" << " vsq: " << fast::squareNorm(state.v) << std::endl;

		}







		alpha = rho / fast::dotProduct<managed_device_vector<T>>(state.r0, state.v);
		if (dv) std::cout << "\t" << " alpha: " << alpha << " = " << rho << "/" << fast::dotProduct(state.r0, state.v) << std::endl;
		{
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__aPlusBetaBKernel << < grid, block >> > (NNZ, THRUST_PTR(state.r), THRUST_PTR(state.v), THRUST_PTR(state.s), -alpha);
		}

		if (dv) std::cout << "\t" << " ssq: " << fast::squareNorm(state.s) << std::endl;

		//Preconditiner b)
		{
			{
				BLOCKS_GRID2D(perBlock2D, NNZ);
				fast::__mulByInverseKernel << < grid, block >> > (NNZ, THRUST_PTR(sys.A.dir[DIR_NONE]), THRUST_PTR(state.s), THRUST_PTR(state.z));
			}
			if (dv) std::cout << "\t" << " zsq: " << fast::squareNorm(state.z) << std::endl;
			{
				BLOCKS3D(perBlock3D, sys.res);
				fast::__matrixVecProductKernel << < numBlocks, block >> > (sys.res, sys.A.getPtr(), THRUST_PTR(state.z), THRUST_PTR(state.t)); //xindex > nnz??
			}
			if (dv) std::cout << "\t" << " tsq: " << fast::squareNorm(state.t) << std::endl;

		}





		T tmp = fast::squareNorm<managed_device_vector<T>>(state.t);


		if (tmp > T(0)) {
			w = fast::dotProduct<managed_device_vector<T>>(state.t, state.s) / tmp;
		}
		else {
			w = T(0);
		}

		if (dv) std::cout << "\t" << " w: " << w << std::endl;

		{
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__ABC_BetaGammaKernel << < grid, block >> > (NNZ, THRUST_PTR(sys.x), THRUST_PTR(state.y), THRUST_PTR(state.z), alpha, w);
		}

		if (dv) std::cout << "\t" << " xsq: " << fast::squareNorm(sys.x) << std::endl;

		{
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__aPlusBetaBKernel << < grid, block >> > (NNZ, THRUST_PTR(state.s), THRUST_PTR(state.t), THRUST_PTR(state.r), -w);
		}

		rsqNorm = fast::squareNorm<managed_device_vector<T>>(state.r);
		if (dv) std::cout << "\t" << " xsq: " << rsqNorm << std::endl;


		if (verbose)
			printf("iter: %u, rsqNorm: %e > %e\n", i, sqrt(rsqNorm / rhs_sqnorm), tol);

		//Diverged
		if (rsqNorm > 10e+10) {
			break;
		}

		++i;

	}

	iterations += i;

	tol_error = sqrt(fast::squareNorm<managed_device_vector<T>>(state.r) / rhs_sqnorm);

	if (verbose) {
		printf("iter: %u, error: %e\n", iterations, tol_error);
	}



	*outError = tol_error;
	*outIterations += iterations;
	return (rsqNorm <= tol2);


}


/*

template <typename T>
void BICG_Output(const LinearSys<T> & sys, CUDA_Volume & out)
{
	BLOCKS3D(8, sys.res);
	if (out.type == TYPE_DOUBLE) {
		vectorToVolume<T,double> << < numBlocks, block >> > (out, THRUST_PTR(sys.x));
	}
	if (out.type == TYPE_FLOAT) {
		vectorToVolume<T, float> << < numBlocks, block >> > (out, THRUST_PTR(sys.x));
	}
}*/



template bool fast::BICG_Solve(LinearSystemDevice<double> & sys, BICGState<double> & state, double tolerance, size_t maxIter, double *outError, size_t * outIterations, bool verbose);
template bool fast::BICG_Solve(LinearSystemDevice<float> & sys, BICGState<float> & state, float tolerance, size_t maxIter, float *outError, size_t * outIterations, bool verbose);


/*
template void BICG_Output(const LinearSys<float> & sys, CUDA_Volume & out);
template void BICG_Output(const LinearSys<double> & sys, CUDA_Volume & out);*/


template struct BICGState<double>;
template struct BICGState<float>;

