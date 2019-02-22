#pragma once
#include <any.hpp>
#include <fastlib/volume/VolumeMeasures.h>
#include <fastlib/volume/VolumeSegmentation.h>

#include "utility.h"

#include <ostream>

#include <map>
#include <unordered_map>

using any = linb::any;



struct RunParams {	
	std::string ID;
	PrimitiveType type;
	std::vector<Dir> dirs;
	fast::DiffusionSolverType solverType;
	fast::Volume * channelPtr;
	fast::Volume * outputPtr = nullptr;
	bool calcRAD;
	bool calcRADBasic = false;
	bool calcTau;
	bool runVerbose;
	bool solverVerbose;
	bool useNU;

	double tolerance;
	size_t maxIter;

	bool useCPU = false;
	int cpuThreads = -1;

	bool oversubscribe = false;

};

using RunRow = std::map<std::string, any>;


template <typename T>
std::vector<RunRow> run(	
	RunParams p
) {
	size_t Nout = p.dirs.size();	
	fast::Volume & c = *p.channelPtr;

	std::map<Dir, RunRow> output;

	for (auto dir : p.dirs) {
		RunRow & o = output[dir];
		o["ID"] = p.ID;
		o["dir"] = dirString(dir);
		o["dimx"] = c.dim().x;
		o["dimy"] = c.dim().y;
		o["dimz"] = c.dim().z;		
		o["precision"] = (p.type == TYPE_DOUBLE) ? std::string("double") : std::string("single");
		o["device"] = p.useCPU ? std::string("CPU") : std::string("GPU");
	}

	if (p.calcRAD || p.calcRADBasic) {
		c.getPtr().createTexture();
	}
	
	//Reactive area density
	std::array<T, 6> radTensor;
	if (p.calcRAD) {
		

		auto t0 = std::chrono::system_clock::now();
		auto ccl = fast::getVolumeCCL(c, 255);
		radTensor = getReactiveAreaDensityTensor<T>(ccl);
		auto t1 = std::chrono::system_clock::now();

		std::chrono::duration<double> dt = t1 - t0;
		double radTime = dt.count();

		for (auto dir : p.dirs) {
			output[dir]["rad"] = radTensor[int(dir)];
			output[dir]["radTime"] = radTime / 6.0f;
		}
	}

	if(p.calcRADBasic){
		auto t0 = std::chrono::system_clock::now();
		float radBasic = fast::getReactiveAreaDensity<T>(c, c.dim(), 0.5f);
		auto t1 = std::chrono::system_clock::now();

		std::chrono::duration<double> dt = t1 - t0;
		double radBasicTime = dt.count();

		for (auto dir : p.dirs) {
			output[dir]["radBasic"] = radBasic;
			output[dir]["radBasicTime"] = radBasicTime;
		}

	}

	//Tortuosity
	if (p.calcTau) {
		TortuosityParams tp;
		tp.verbose = p.solverVerbose;
		tp.coeffs = { 1.0, 0.001 };
		tp.maxIter = p.maxIter;
		tp.tolerance = p.tolerance;
		tp.porosity = getPorosity<double>(c);
		tp.porosityPrecomputed = true;
		tp.useNonUniform = p.useNU;
		tp.oversubscribeGPUMemory = p.oversubscribe;
		tp.cpuThreads = p.cpuThreads;
		tp.onDevice = !p.useCPU;

		for (auto dir : p.dirs) {			
			RunRow & o = output[dir];
			tp.dir = dir;

			if (p.runVerbose) {
				std::cout << "Tau Direction " << tp.dir << std::endl;
			}
			//Calculate tortuosity 
			auto t0 = std::chrono::system_clock::now();
			size_t iter = size_t(-1); 
			T tau = getTortuosity<T>(c, tp, p.solverType, p.outputPtr, &iter);
			auto t1 = std::chrono::system_clock::now();
			std::chrono::duration<double> dt = t1 - t0;
			o["tau"] = tau;
			o["tauTime"] = dt.count();		
			o["tauSolver"] = solverString(p.solverType);
			o["tauNU"] = tp.useNonUniform;
			o["tauTolerance"] = tp.tolerance;
			o["porosity"] = tp.porosity;
			o["tauIter"] = int(iter);

			if (p.runVerbose) {
				std::cout << "Elapsed: " << dt.count() << "s (" << dt.count() / 60.0 << "m)" << std::endl;
			}
		}
	}
	

	std::vector<RunRow> vecOut;
	for (auto & it : output)
		vecOut.push_back(std::move(it.second));

	return std::move(vecOut);

/*
	/ *
	Output result
	* /
	{
		bool isNewFile = true;

		//Check if output file exists
		if (argOutput) {
			std::ifstream f(argOutput.Get());
			if (f.is_open())
				isNewFile = false;
		}

		//If output file, open it
		std::ofstream outFile;

		if (argOutput) {
			outFile.open(argOutput.Get(), std::ios::app);
		}
		//Choose output stream
		std::ostream & os = (outFile.is_open()) ? outFile : std::cout;



		//Header 
		if (isNewFile) {
			os << "path,porosity,dir,tolerance,tau,";
			if (argRad.Get()) {
				os << "rad,";
				/ *os << "rad_X_POS,rad_X_NEG,";
				os << "rad_Y_POS,rad_Y_NEG,";
				os << "rad_Z_POS,rad_Z_NEG,";* /
			}
			os << "t,dimx,dimy,dimz,solver" << '\n';
		}

		for (auto i = 0; i < taus.size(); i++) {
			os << "'" << name << "'" << ",\t";

			os << tp.porosity << ",\t";

			os << dirString(dirs[i]) << ",\t";

			os << tp.tolerance << ",\t";

			os << taus[i] << ",\t";

			if (argRad.Get()) {
				//for (auto i = 0; i < 6; i++)
				os << radTensor[i] << ",\t";
			}

			//double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
			os << times[i] << ",\t";
			//os << avgTime << ",\t";

			auto dim = c.dim();
			os << dim.x << ",\t" << dim.y << ",\t" << dim.z;


			os << ",\t" << argSolver.Get();

			os << "\n";
		}
	}

	return true;*/

}



double sphereAnalytical(const std::string & path) {
	//auto inputPath = fs::path(path);

	if (checkExtension(path,"sph")) {
		std::cerr << "Is not .sph file" << std::endl;
		return 0;
	}


	std::ifstream f(path);
	auto spheres = fast::loadSpheres(f);

	fast::GeneratorSphereParams p;
	p.withinBounds = true;

	return fast::spheresAnalyticTortuosity(p, spheres);


}