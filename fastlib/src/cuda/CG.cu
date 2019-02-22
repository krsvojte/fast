#include "CG.cuh"




template <typename T>
bool CGState<T>::resize(size_t N, bool precondition)
{
	r.resize(N);
	p.resize(N);
	ap.resize(N);
	if (precondition)
		z.resize(N);
	return true;
}




template <typename T>
bool fast::CG_Solve(LinearSystemDevice<T> & sys, CGState<T> & state, T tolerance, size_t maxIter, T *outError, size_t * outIterations, bool verbose)
{

	const size_t maxIters = maxIter;
	const int perBlock2D = 1024;

	bool precondition = (state.z.size() > 0);

	size_t iterations = 0;
	size_t i = 0;
	size_t NNZ = sys.NNZ;

	T tol = tolerance;
	T tol_error = 1.0;



	fast::LinearSys_Residual<T>(sys, state.r);
	T rhs_sqnorm = fast::squareNorm(sys.b);

	if (rhs_sqnorm == 0) {
		thrust::fill(sys.x.begin(), sys.x.end(), T(0));
		return T(0);
	}

	T tol2 = tol*tol*rhs_sqnorm;

	T alpha = 0;
	T beta = 0;
	T rsqNorm = 0;
	rsqNorm = fast::squareNorm<managed_device_vector<T>>(state.r);

	T dotZ_R = 0;



	if (precondition) {
		// z= M^-1*r_0
		BLOCKS_GRID2D(perBlock2D, NNZ);
		fast::__mulByInverseKernel << < grid, block >> > (NNZ, THRUST_PTR(sys.A.dir[DIR_NONE]), THRUST_PTR(state.r), THRUST_PTR(state.z));
		state.p = state.z;
		dotZ_R = fast::dotProduct<managed_device_vector<T>>(state.r, state.z);
	}
	else {
		state.p = state.r;
	}

	while (rsqNorm > tol2 && i < maxIters) {

		//ap = A*p
		{
			BLOCKS3D(8, sys.res);
			fast::__matrixVecProductKernel << < numBlocks, block >> > (sys.res, sys.A.getPtr(), THRUST_PTR(state.p), THRUST_PTR(state.ap));
		}


		T dotP_AP = fast::dotProduct<managed_device_vector<T>>(state.p, state.ap);

		if (precondition)
			//alpha = (z dot r) / (p' * A * p)	
			alpha = dotZ_R / dotP_AP;
		else
			//alpha = (r dot r) / (p' * A * p)	
			alpha = rsqNorm / dotP_AP;


		//x += alpha*p
		{
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__aAddBetaBKernel << < grid, block >> > (
				NNZ, THRUST_PTR(sys.x), THRUST_PTR(state.p), alpha
				);
		}


		//r += -alpha*ap
		{
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__aAddBetaBKernel << < grid, block >> > (
				NNZ, THRUST_PTR(state.r), THRUST_PTR(state.ap), -alpha
				);
		}

		

		//If new r small enough
		T oldRsqnorm = rsqNorm;
		rsqNorm = fast::squareNorm<managed_device_vector<T>>(state.r);
		if (rsqNorm <= tol2)
			break;


		if (precondition) {
			// z= M^-1*r_i
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__mulByInverseKernel << < grid, block >> > (NNZ, THRUST_PTR(sys.A.dir[DIR_NONE]), THRUST_PTR(state.r), THRUST_PTR(state.z));
		}


		if (precondition) {
			T oldDotZ_R = dotZ_R;
			dotZ_R = fast::dotProduct<managed_device_vector<T>>(state.r, state.z);
			beta = dotZ_R / oldDotZ_R;
		}
		else {
			beta = rsqNorm / oldRsqnorm;
		}

		// p = p*beta + r
		if (precondition) {
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__aAddAlphaBKernel << < grid, block >> > (
				NNZ, THRUST_PTR(state.p), THRUST_PTR(state.z), beta
				);
		}
		else
		{
			BLOCKS_GRID2D(perBlock2D, NNZ);
			fast::__aAddAlphaBKernel << < grid, block >> > (
				NNZ, THRUST_PTR(state.p), THRUST_PTR(state.r), beta
				);
		}

		++i;



		if (/*i % 100 == 0 &&*/ verbose)
			printf("iter: %u, rsqNorm: %e > %e\n", i, sqrt(rsqNorm / rhs_sqnorm), tol);

		//Diverged
		if (rsqNorm > 10e+10) {
			break;
		}
	}


	iterations += i;

	tol_error = sqrt(fast::squareNorm<managed_device_vector<T>>(state.r) / rhs_sqnorm);

	if (verbose)
		printf("iter: %u, error: %e\n", iterations, tol_error);


	*outError = tol_error;
	*outIterations = iterations;
	return (rsqNorm <= tol2);


}


/*


template <typename T>
void CG_Output(const LinearSys<T> & sys, CUDA_Volume & out)
{
	BLOCKS3D(8, sys.res);
	if (out.type == TYPE_DOUBLE) {
		vectorToVolume<T, double> << < numBlocks, block >> > (out, THRUST_PTR(sys.x));
	}
	if (out.type == TYPE_FLOAT) {
		vectorToVolume<T, float> << < numBlocks, block >> > (out, THRUST_PTR(sys.x));
	}
}
*/






template bool fast::CG_Solve(LinearSystemDevice<double> & sys, CGState<double> & state, double tolerance, size_t maxIter, double *outError, size_t * outIterations, bool verbose);
template bool fast::CG_Solve(LinearSystemDevice<float> & sys, CGState<float> & state, float tolerance, size_t maxIter, float *outError, size_t * outIterations, bool verbose);

/*
template void CG_Output(const LinearSys<float> & sys, CUDA_Volume & out);
template void CG_Output(const LinearSys<double> & sys, CUDA_Volume & out);*/

template struct CGState<double>;
template struct CGState<float>;