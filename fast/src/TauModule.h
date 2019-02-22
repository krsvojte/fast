#pragma once
#include "Module.h"
#include "VolumeIOModule.h"
#include "CSVOutputModule.h"
#include "utility.h"



class TauModule : public Module {

public:
	TauModule(args::Group & group) :
		_inputModule(group),
		_outputModule(group),
		_argDir(group, "string", "Direction (x|y|z)|all|pos|neg", { 'd', "dir" }, "x-"),
		_argTol(group, "tolerance", "Tolerance 1e-{tol}", { "tol" }, 6),
		_argMaxIter(group, "maxIterations", "Max Iterations", { "iter" }, 10000),
		_argOver(group, "oversubscribe", "Oversubsribe GPU memory (unix only)", { "oversubscribe", "over" }),
		_argPrecision(group, "precision", "Precision (f|d) Float or Double", { 'p', "precision" }, "double"),
		_argCPU(group, "cpu", "Use CPU Solver", { "cpu" }),
		_argCPUThreads(group, "cputhreads", "CPU Solver Threads: -1 uses all available cores.", { 'j', "cputhreads" }, -1),
		_argVerbose(group, "v", "Verbose", { 'v', "verbose" }),
		_argSolver(group, "string", "Solver (CG|BICGSTAB)", { "solver" }, "CG")
	{
	
	}

	virtual void prepare() override {
		_inputModule.prepare();
		_outputModule.prepare();

	}

	static std::vector<RunRow> getTau(
		const fast::Volume & volume,
		const std::string & name,
		bool CPU,
		int CPUThreads,
		bool verbose,
		const std::vector<Dir> & dirs,
		PrimitiveType type,
		DiffusionSolverType solverType,
		int maxIter,
		double tolerance,
		bool oversubscribe

	){

		std::vector<RunRow> res;
		
		Volume * outputPtr = nullptr;


		RunRow o;
		o["ID"] = name;
		o["precision"] = (type == TYPE_DOUBLE) ? std::string("double") : std::string("single");
		o["device"] = CPU ? std::string("CPU") : std::string("GPU");
		o["dimx"] = volume.dim().x;
		o["dimy"] = volume.dim().y;
		o["dimz"] = volume.dim().z;

		float porosity = getPorosity<double>(volume);

		for (auto dir : dirs) {

			o["dir"] = dirString(dir);

			TortuosityParams tp;
			tp.verbose = verbose;
			tp.coeffs = { 1.0, 0.001 };
			tp.maxIter = maxIter;
			tp.tolerance = tolerance;
			tp.porosity = porosity;
			tp.porosityPrecomputed = true;
			tp.useNonUniform = false;
			tp.oversubscribeGPUMemory = oversubscribe;
			tp.cpuThreads = CPUThreads;
			tp.onDevice = !(CPU);
			tp.dir = dir;


			if (verbose) {
				std::cout << "Tau Direction " << tp.dir << std::endl;
			}

			//Calculate tortuosity 
			auto t0 = std::chrono::system_clock::now();
			size_t iter = size_t(-1);

			double tau = 0.0;
			if (type == TYPE_DOUBLE) {
				tau = getTortuosity<double>(volume, tp, solverType, outputPtr, &iter);
			}
			else {
				tau = static_cast<double>(getTortuosity<float>(volume, tp, solverType, outputPtr, &iter));
			}

			auto t1 = std::chrono::system_clock::now();
			std::chrono::duration<double> dt = t1 - t0;

			o["tau"] = tau;
			o["tauTime"] = dt.count();
			o["tauSolver"] = solverString(solverType);
			o["tauNU"] = tp.useNonUniform;
			o["tauTolerance"] = tp.tolerance;
			o["porosity"] = tp.porosity;
			o["tauIter"] = int(iter);

			if (verbose) {
				std::cout << "Elapsed: " << dt.count() << "s (" << dt.count() / 60.0 << "m)" << std::endl;
			}

			res.push_back(o);			
		}

		return res;

	}

	virtual void execute() override {


		std::vector<Dir> dirs = getDirs(_argDir.Get());

		PrimitiveType type = (_argPrecision.Get() == "float" ||
			_argPrecision.Get() == "single" ||
			_argPrecision.Get() == "f" ||
			_argPrecision.Get() == "s")
			? TYPE_FLOAT : TYPE_DOUBLE;

		DiffusionSolverType solverType = DSOLVER_CG;
		if (_argSolver.Get() == "BICGSTAB")
			solverType = DSOLVER_BICGSTAB;
		else if (_argSolver.Get() == "CG")
			solverType = DSOLVER_CG;
		
		double tolerance = double(pow(10.0, -_argTol.Get()));	
		
	
		while (true) {
			auto volume = _inputModule.getNext();
			if (!volume) break;
			
			auto rows  = getTau(
				*volume,
				_inputModule.getVolumeName(volume.get()),
				_argCPU.Get(),
				_argCPUThreads.Get(),
				_argVerbose.Get(),
				dirs,
				type,
				solverType, 
				_argMaxIter.Get(), 
				tolerance, 
				_argOver.Get()
			);
			
			for (auto & row : rows) {
				_outputModule.addRunRow(row);
			}

		}
	
	}

private:
	
	VolumeInputModule _inputModule;
	CSVOutputModule _outputModule;

	args::ValueFlag<std::string> _argDir;

	args::ValueFlag<std::string> _argSolver;

	args::ValueFlag<int> _argTol;
	args::ValueFlag<int> _argMaxIter;
	args::Flag _argOver;
	args::ValueFlag<std::string> _argPrecision;

	args::Flag _argCPU;
	args::ValueFlag<int> _argCPUThreads;
	args::Flag _argVerbose;

	

	
};



