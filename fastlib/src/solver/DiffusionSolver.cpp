#ifdef _deprecated
#include "solver/DiffusionSolver.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>


#include <omp.h>
#include <stack>

#include "utility/PrimitiveTypes.h"


#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>


//#define DS_LINSYS_TO_FILE

using namespace fast;



enum NeighType {
	NODE, 
	DIRICHLET,
	VON_NEUMANN
};

#define NO_NODE (size_t(0)-1)

struct Neigh {
	union {
		int index; //if node type
		float value; //if vonneumman -> add to diagonal, if dirichlet -> subtract from b
	};
	NeighType type;
};

struct Node {
	Node(ivec3 pos_) : pos(pos_) {}
	ivec3 pos;  //for debug
	Neigh neigh[6];
};




void buildNodeList(std::vector<Node> & nodeList, std::vector<size_t> & indices, const Volume & c, size_t nodeIndex_) {

	const auto & dim = c.dim();
	const size_t stride[3] = { 1, static_cast<size_t>(dim.x), static_cast<size_t>(dim.x*dim.y) };
	const uchar * cdata = (uchar *)c.getPtr().getCPU();
	//todo add bound.cond here
	/*if (curPos.x < 0 || curPos.y < 0 || curPos.z < 0 ||
		curPos.x >= dim.x || curPos.y >= dim.y || curPos.z >= dim.z
		) return;*/

	
	
	

	std::stack<ivec3> stackNodePos;
	

	for (auto y = 0; y < dim.y; y++) {
		for (auto z = 0; z < dim.z; z++) {
			stackNodePos.push({ 0,y,z });
		}
	}
	while (!stackNodePos.empty()) {

		ivec3 nodePos = stackNodePos.top();		
		stackNodePos.pop();

		auto i = linearIndex(dim, nodePos);

	
		

		//Node at d1, skip
		if (indices[i] == NO_NODE) {
			if(cdata[i] != 0)
				continue;

			nodeList.push_back(Node(nodePos));
			indices[i] = nodeList.size() - 1;
		}


		assert(cdata[i] == 0);

		size_t nodeIndex = indices[i];
		

		auto * n = &nodeList[nodeIndex];
		

		//Visited
		//if (indices[i] != NO_NODE) return;

		const float conc[2] = { 0.0f,1.0f };
		const int dirichletDir = 0; //x dir

		const float highConc = 1.0f;
		const float lowConc = 0.0f;
		const float d0 = 0.001f;

		static const ivec3 dirVecs[6] = {
			ivec3(1,0,0),
			ivec3(-1,0,0),
			ivec3(0,1,0),
			ivec3(0,-1,0),
			ivec3(0,0,1),
			ivec3(0,0,-1)
		};

		for (auto k = 0; k < (3); k++) {
			for (auto dir = 0; dir < 2; dir++) {
				const int neighDir = 2 * k + dir;


				//Box domain conditions
				if ((dir == 0 && n->pos[k] == dim[k] - 1) ||
					(dir == 1 && n->pos[k] == 0)) {
					//Dirichlet  
					if (k == dirichletDir) {
						n->neigh[neighDir].type = DIRICHLET;
						n->neigh[neighDir].value = conc[dir]; //x1 or x0 conc
					}
					else {
						n->neigh[neighDir].type = VON_NEUMANN;
						n->neigh[neighDir].value = d0;
					}
				}
				//
				else {

					const ivec3 dirVec = dirVecs[2 * k + dir];
					const int thisStride = -(dir * 2 - 1) * int(stride[k]); // + or -, stride in k dim

					if (cdata[i + thisStride] == 0) {

						n->neigh[neighDir].type = NODE;

						int ni = static_cast<int>(linearIndex(dim, n->pos + dirVec));
						if (indices[ni] == NO_NODE) {
							nodeList.push_back(Node(n->pos + dirVec));
							indices[ni] = nodeList.size() - 1;
							n = &nodeList[nodeIndex]; //refresh pointer if realloc happened
							n->neigh[neighDir].index = static_cast<int>(indices[ni]);
							stackNodePos.push(n->pos + dirVec);
						}
						else {
							n->neigh[neighDir].index = static_cast<int>(indices[ni]);
						}


					}
					else {
						n->neigh[neighDir].type = VON_NEUMANN;
						n->neigh[neighDir].value = d0;
					}
				}
			}

		}

	}

/*

	if (n.pos.x == 0) {
		n.neigh[X_NEG].type = DIRICHLET;
		n.neigh[X_NEG].value = highConc;
	}
	else {
		if (cdata[i - stride[0]] == 0){
			int ni = linearIndex(dim, n.pos + ivec3(-1, 0, 0));
			if (indices[ni] == NO_NODE) {
				nodeList.push_back(Node(n.pos + ivec3(-1, 0, 0)));
				indices[ni] = nodeList.size() - 1;
				buildNodeList(nodeList, indices, c, nodeList.back());
			}
			n.neigh[X_NEG].index = indices[ni];
			n.neigh[X_NEG].type = NODE;
		}
		else {
			n.neigh[X_NEG].type = VON_NEUMANN;
			n.neigh[X_NEG].value = d0;
		}
	}

	if (n.pos.x == dim.x - 1) {
		n.neigh[X_POS].type = DIRICHLET;
		n.neigh[X_POS].value = lowConc;
	}
	else {
		if (cdata[i + stride[0]] == 0) {
			int ni = linearIndex(dim, n.pos + ivec3(1, 0, 0));
			if (indices[ni] == NO_NODE) {
				nodeList.push_back(Node(n.pos + ivec3(1, 0, 0)));
				indices[ni] = nodeList.size() - 1;
				buildNodeList(nodeList, indices, c, nodeList.back());
			}
			n.neigh[X_POS].index = indices[ni];
			n.neigh[X_POS].type = NODE;
		}
		else {
			n.neigh[X_POS].type = VON_NEUMANN;
			n.neigh[X_POS].value = d0;
		}
	}
*/


/*

	if (n.pos.x == dim.x - 1) {
		n.neigh[X_NEG].type = DIRICHLET;
		n.neigh[X_NEG].value = lowConc;
	}
	else {
		

	}*/

	





}


template <typename T>
DiffusionSolver<T>::DiffusionSolver(bool verbose)
	: _verbose(verbose)	 
{
	
}

template <typename T>
DiffusionSolver<T>::~DiffusionSolver()
{

}

template <typename T>
bool DiffusionSolver<T>::prepare(const Volume & volChannel, Dir dir, T d0, T d1, bool preserveAspectRatio) {

	using vec3 = glm::tvec3<T, glm::highp>;

	auto dim = volChannel.dim();
	assert(volChannel.type() == TYPE_UCHAR);
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);
	uint dirPrimary = getDirIndex(dir);
	uint dirSecondary[2] = { (dirPrimary + 1) % 3, (dirPrimary + 2) % 3 };
	const T highConc = 1.0f;
	const T lowConc = 0.0f;
	const T concetrationBegin = (getDirSgn(dir) == 1) ? highConc : lowConc;
	const T concetrationEnd = (getDirSgn(dir) == 1) ? lowConc : highConc;

	size_t N = dim.x * dim.y * dim.z;
	size_t M = N;

	_rhs.resize(M);
	_x.resize(N);


	//D data
	const uchar * D = (uchar *)volChannel.getPtr().getCPU();

	//D getter
	const auto getD = [dim, D, d0, d1](ivec3 pos) {
		return (D[linearIndex(dim, pos)] == 0) ? d0 : d1;
	};

	const auto sample = [&getD, dim, D, d0, d1](ivec3 pos, Dir dir) {
		const int k = getDirIndex(dir);
		assert(pos.x >= 0 && pos.y >= 0 && pos.z >= 0);
		assert(pos.x < dim.x && pos.y < dim.y && pos.z < dim.z);

		ivec3 newPos = pos;
		int sgn = getDirSgn(dir);
		if (pos[k] + sgn < 0 || pos[k] + sgn >= dim[k]) {
			//do nothing
		}
		else {
			newPos[k] += getDirSgn(dir);
		}
		return getD(newPos);
	};


	//Default spacing

	

	vec3 cellDim = { T(1) / dim.x , T(1) / (dim.y),T(1) / (dim.z) };


	if (preserveAspectRatio) {

		T maxDim = static_cast<T>(std::max(dim[0], std::max(dim[1], dim[2])));		
		cellDim = { T(1) / maxDim,T(1) / maxDim ,T(1) / maxDim };		
	}
	


	vec3 faceArea = {
		cellDim.y * cellDim.z,
		cellDim.x * cellDim.z,
		cellDim.x * cellDim.y,
	};

	

	size_t d0Count = 0;
	size_t d1Count = 0;

	_A.resize(M, N);
	_A.reserve(Eigen::VectorXi::Constant(N, 7));
	

	struct Coeff {
		Dir dir;
		T val;
		signed long col;		
		bool useInMatrix;
		bool operator < (const Coeff & b) const { return this->col < b.col; }
	};



	ivec3 stride = { 1, dim.x, dim.x*dim.y };

	for (auto z = 0; z < dim.z; z++) {
		for (auto y = 0; y < dim.y; y++) {
			for (auto x = 0; x < dim.x; x++) {


				const ivec3 ipos = { x,y,z };
				auto i = linearIndex(dim, ipos);

				T Di = getD(ipos);
				if (Di == d0)
					d0Count++;
				else
					d1Count++;

				auto Dvec = vec3(Di);

				auto Dneg = (vec3(
					sample(ipos, X_NEG),
					sample(ipos, Y_NEG),
					sample(ipos, Z_NEG)
				) + vec3(Dvec)) * T(0.5);

				auto Dpos = (vec3(
					sample(ipos, X_POS),
					sample(ipos, Y_POS),
					sample(ipos, Z_POS)
				) + vec3(Dvec)) * T(0.5);


				std::array<Coeff, 7> coeffs;
				T rhs = T(0);
				for (auto k = 0; k <= DIR_NONE; k++) {
					coeffs[k].dir = Dir(k);
					coeffs[k].val = T(0);
					coeffs[k].useInMatrix = true;					
				}
				coeffs[DIR_NONE].col = static_cast<signed long>(i);

				

				//Calculate coeffs for all except diagonal
				for (auto j = 0; j < DIR_NONE; j++) {					
					auto & c = coeffs[j];
					auto k = getDirIndex(c.dir);
					auto sgn = getDirSgn(c.dir);
					auto Dface = (sgn == -1) ? Dneg[k] : Dpos[k];
					
					/*
						Boundary defaults
						1. half size cell
						2. von neumann zero grad
					*/
					auto cellDist = cellDim;
					c.useInMatrix = true;
					if (ipos[k] == 0 && sgn == -1) {
						cellDist[k] = cellDim[k] * T(0.5);
						c.useInMatrix = false;
					}
					if (ipos[k] == dim[k] - 1 && sgn == 1) {
						cellDist[k] = cellDim[k] * T(0.5);
						c.useInMatrix = false;						
					}

					c.val = (Dface * faceArea[k]) / cellDist[k];								
					c.col = static_cast<signed long>(i + sgn * stride[k]);

					//Add to diagonal
					if(c.useInMatrix || k == dirPrimary)
						coeffs[DIR_NONE].val -= c.val;								
				}

				if (ipos[dirPrimary] == 0) {
					Dir dir = getDir(dirPrimary, -1);
					rhs -= coeffs[dir].val * concetrationBegin;
				}
				else if (ipos[dirPrimary] == dim[dirPrimary]-1) {
					Dir dir = getDir(dirPrimary, 1);
					rhs -= coeffs[dir].val * concetrationEnd;
				}

				//Matrix coefficients
				std::sort(coeffs.begin(), coeffs.end());
				for (auto & c : coeffs) {
					if (c.useInMatrix) {
						_A.insert(i, c.col) = c.val;
					}
				}

				//Right hand side
				_rhs[i] = rhs;

				//initial guess
				if (getDirSgn(dir) == 1)
					_x[i] = 1.0f - (ipos[dirPrimary] / T(dim[dirPrimary] + 1));
				else
					_x[i] = (ipos[dirPrimary] / T(dim[dirPrimary] + 1));

			}
		}
	}


	_porosity = T(d0Count) / T(d0Count + d1Count);


#ifdef DS_LINSYS_TO_FILE

	{
		std::ofstream f("A.dat");

		for (auto i = 0; i < _A.rows(); i++) {
			for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(_A, i); it; ++it) {
				auto  j = it.col();
				f << i << " " << j << " " << it.value() << "\n";
			}

			if (_A.rows() < 100 || i % (_A.rows() / 100))
				f.flush();
		}
	}

	{
		std::ofstream f("B.txt");
		for (auto i = 0; i < _rhs.size(); i++) {
			f << _rhs[i] << "\n";
			if (_rhs.size() < 100 || i % (_rhs.size() / 100))
				f.flush();
		}

	}



#endif


	



	_A.makeCompressed();



	if(false){
		//https://www.sciencedirect.com/science/article/pii/S0377042710002979
		size_t rows = _A.rows();
		size_t cols = _A.cols();
		Eigen::SparseMatrix<T, Eigen::RowMajor> D;
		D.resize(rows, cols);
		D.reserve(Eigen::VectorXi::Constant(rows, 1));

		for (auto i = 0; i < rows; i++) {

			T sum = 0;
			int n = 0;
			for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(_A, i); it; ++it) {
				T val = it.value();
				sum += val*val;
				n++;
			}
			T norm = sqrt(sum);

			T d = static_cast<T>(1.0) / norm;

			for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(_A, i); it; ++it) {
				auto  j = it.col();
				it.valueRef() *= d;
			}

			_rhs(i) *= d;

		}
	}





	_solver.compute(_A);

	return true;
}





template <typename T>
T DiffusionSolver<T>::solve(T tolerance, size_t maxIterations, size_t iterPerStep)
{
	
	iterPerStep = std::min(iterPerStep, maxIterations);
	_solver.setTolerance(tolerance);
	_solver.setMaxIterations(iterPerStep);

	T err = std::numeric_limits<T>::max();
		
//	_x.setZero();

	for (size_t i = 0; i < maxIterations; i += iterPerStep) {

		_x = _solver.solveWithGuess(_rhs, _x);
		
		err = _solver.error();

		if (_verbose) {
			std::cout << "i:" << i << ", estimated error: " << err << ", ";
		
			switch (_solver.info()) {
			case Eigen::ComputationInfo::Success:
				std::cout << "Success"; break;
			case Eigen::ComputationInfo::NumericalIssue:
				std::cout << "NumericalIssue"; break;
			case Eigen::ComputationInfo::NoConvergence:
				std::cout << "NoConvergence"; break;
			case Eigen::ComputationInfo::InvalidInput:
				std::cout << "InvalidInput"; break;
			}

			std::cout << std::endl;
		}

		if (err <= tolerance)
			break;		
	}


	return err;

}


template <typename T>
bool DiffusionSolver<T>::resultToVolume(Volume & vol)
{

	void * destPtr = vol.getPtr().getCPU();

	//Copy directly if same type
	if ((std::is_same<float, T>::value && vol.type() == TYPE_FLOAT) ||
		(std::is_same<double, T>::value && vol.type() == TYPE_DOUBLE)) {
		memcpy(destPtr, _x.data(), _x.size() * sizeof(T));
	}
	else {
		if (vol.type() == TYPE_FLOAT) {
			Eigen::Matrix<float, Eigen::Dynamic, 1> tmpX = _x.template cast<float>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(float));
		}
		else if (vol.type() == TYPE_DOUBLE) {
			Eigen::Matrix<double, Eigen::Dynamic, 1> tmpX = _x.template cast<double>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(double));
		}
		else {
			return false;
		}
	}	
	
	return true;


}












template <typename T>
FAST_EXPORT bool DiffusionSolver<T>::solveWithoutParticles(
	Volume & volChannel, 
	Volume * outVolume, 
	float d0,
	float d1,
	float tolerance
)
{
#ifdef DS_USEGPU
	assert(false); // CPU only
#endif

	

	const auto & c = volChannel;
	auto dim = c.dim();

	if (_verbose)
		std::cout << "DIM " << dim.x << ", " << dim.y << ", " << dim.z << " nnz:" << dim.x*dim.y*dim.z / (1024 * 1024.0f) << "M" << std::endl;


	const float highConc = 1.0f;
	const float lowConc = 0.0f;
	//const float d0 = 0.001f;

	//Max dimensions
	size_t maxN = dim.x * dim.y * dim.z; //cols
	size_t maxM = maxN; //rows	


	auto start0 = std::chrono::system_clock::now();

	std::vector<Node> nodeList;
	nodeList.reserve(maxN); 
	std::vector<size_t> indices(volChannel.dim().x*volChannel.dim().y*volChannel.dim().z, 0-1);
	buildNodeList(nodeList, indices, volChannel, 1);


	float porosity = 0.0f;
	{
		const uchar * cdata = (uchar *)volChannel.getPtr().getCPU();
		size_t cnt = 0;
		for (auto i = 0; i < volChannel.getPtr().byteSize() / sizeof(uchar); i++) {
			if (cdata[i] == 0)
				cnt++;
		}
		porosity = (cnt / float(maxN));
	}
	
	if (_verbose)
	{		
		std::cout << "occupancy (d0, direct): " << 100.0f * porosity << std::endl;
	}

	if (_verbose)
		std::cout << "occupancy (d0): " << 100.0f * nodeList.size() / float(dim.x*dim.y*dim.z) << "%  (" << nodeList.size() / (1024 * 1024.0f) << "M)" << std::endl;

	auto end0= std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds0 = end0 - start0;
	if (_verbose) {
		std::cout << "prep elapsed time: " << elapsed_seconds0.count() << "s\n";
	}



	////////////////////////////

	size_t M = nodeList.size();
	size_t N = M;

	const vec3 h = { 1.0f / (dim.x + 1), 1.0f / (dim.y + 1), 1.0f / (dim.z + 1) };
	const vec3 invH = { 1.0f / h.x, 1.0f / h.y, 1.0f / h.z };
	const vec3 invH2 = { invH.x*invH.x,invH.y*invH.y,invH.z*invH.z };


	std::vector<Eigen::Triplet<T>> triplets;
	
	Eigen::Matrix<T, Eigen::Dynamic, 1> b(M);
	b.setZero();
	Eigen::Matrix<T, Eigen::Dynamic, 1> x(N);

	size_t row = 0;
	for (auto & n : nodeList) {

		auto Dcur = vec3(d0) * invH2; // vec x vec product
		auto Dpos = Dcur;
		auto Dneg = Dcur;

		auto diagVal = -(Dpos.x + Dpos.y + Dpos.z + Dneg.x + Dneg.y + Dneg.z);
		float bval = 0.0f;

		int k = 0;
		for (auto & neigh : n.neigh) {
			float dval = (k % 2 == 0) ? Dpos[k / 2] : Dneg[k / 2];

			if (neigh.type == NODE) {
				triplets.push_back(Eigen::Triplet<T>(static_cast<int>(row), neigh.index,
					dval
					));
			}
			else if (neigh.type == DIRICHLET) {
				bval -= d0 * neigh.value * invH2[k / 3];
			}
			else if (neigh.type == VON_NEUMANN) {
				diagVal += dval;
			}
			k++;
		}

		b[row] = bval;

		//initial guess
		x[row] = 1.0f - (n.pos.x / float(dim.x + 1));
		
		triplets.push_back(Eigen::Triplet<T>(static_cast<int>(row), row, diagVal));
		
		row++;
	}

	auto start = std::chrono::system_clock::now();
	Eigen::SparseMatrix<T, Eigen::RowMajor> A(M, N);
	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();


	omp_set_num_threads(8);
	Eigen::setNbThreads(8);

	/*Eigen::VectorXf res;
	for (auto i = 0; i < 1024; i++) {
		jacobi(A, b, x);
		res = A*b - x;
		std::cout << "jacobi res: " << res.mean() << std::endl;
	}*/

	Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>> stab;	

	
	stab.setTolerance(tolerance);
	stab.compute(A);
	

	const int maxIter = 3000;
	const int iterPerStep = 100;

	stab.setMaxIterations(iterPerStep);

	int i = 0;
	for (i = 0; i < maxIter; i += iterPerStep) {
		x = stab.solveWithGuess(b, x);		
		//iter += stab.iterations();
		float er = static_cast<float>(stab.error());
		if (_verbose) {			
			std::cout << "i:" << i << ", estimated error: " << er << std::endl;
		}
		if (er <= tolerance)
			break;
	}	
	

	_iterations = static_cast<uint>(stab.iterations());

	auto end = std::chrono::system_clock::now();


	if (_verbose) {
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "#iterations:     " << stab.iterations() << std::endl;
		std::cout << "estimated error: " << stab.error() << std::endl;
		std::cout << "solve host elapsed time: " << elapsed_seconds.count() << "s\n";
		std::cout << "host avg conc: " << x.mean() << std::endl;
		std::cout << "tolerance " << tolerance << std::endl;

	}

	//Convert solution to volume
	{
		float * concData = (float *)outVolume->getPtr().getCPU();
		const uchar * cdata = (uchar *)volChannel.getPtr().getCPU();
		outVolume->clear();
		int nodeIndex = 0;
		for (auto & n : nodeList) {
			auto i = linearIndex(dim, n.pos);
			concData[i] = static_cast<float>(x[nodeIndex]);			
			nodeIndex++;

		}
	}

	return true;
}

template <typename T>
FAST_EXPORT T fast::DiffusionSolver<T>::tortuosity(const Volume & mask, Dir dir)
{
	
	
	assert(mask.type() == TYPE_UCHAR || mask.type() == TYPE_CHAR);
	const auto dim = mask.dim();		

	const T * concData = _x.data();
	const uchar * cdata = (uchar *)mask.getPtr().getCPU();	

	const int primaryDim = getDirIndex(dir);
	const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };

	vec3 cellDim = { T(1) / dim.x , T(1) / (dim.y),T(1) / (dim.z) };
	
	bool preserveAspectRatio = true;
	if (preserveAspectRatio) {

		T maxDim = static_cast<T>(std::max(dim[0], std::max(dim[1], dim[2])));
		cellDim = { T(1) / maxDim,T(1) / maxDim ,T(1) / maxDim };
	}
	

	int n = dim[secondaryDims[0]] * dim[secondaryDims[1]];
	int k = (getDirSgn(dir) == -1) ? 0 : dim[primaryDim] - 1;	

	/*
		Calculate average in low concetration plane
	*/
	bool zeroOutPart = true;
	std::vector<T> isums(dim[secondaryDims[0]], T(0));
	
	#pragma omp parallel for
	for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
		T jsum = T(0);
		for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
			ivec3 pos;
			pos[primaryDim] = k;
			pos[secondaryDims[0]] = i;
			pos[secondaryDims[1]] = j;

			if(zeroOutPart && cdata[linearIndex(dim, pos)] == 0)
				jsum += concData[linearIndex(dim, pos)];			
		}
		isums[i] = jsum;
	}
	
	T sum = std::accumulate(isums.begin(),isums.end(),T(0));

	double dc = sum / n;
	double dx = cellDim[primaryDim];
	double tau = /*dx * */_porosity / (dc * /*dx **/ dim[primaryDim] * 2);

	if (_verbose) {
		std::cout << "dc: " << dc << std::endl;
		std::cout << "porosity: " << _porosity << std::endl;
		std::cout << "tau: " << tau << std::endl;
	}	

	return static_cast<T>(tau);

	

	
}


namespace fast{
template class DiffusionSolver<float>;
template class DiffusionSolver<double>;
}

#endif