#pragma once
#include "Module.h"

#include <fastlib/optimization/SDFPacking.h>
#include <tuple>
#include <set>


std::istream& operator>>(std::istream& is, fast::vec3 & v)
{	
	is >> v.x >> v.y >> v.z;	
	return is;
}


class PackingModule : public Module {

public:
	PackingModule(args::Group & group) :
		_argTargetEps(group, "Target Porosity", "Target Porosity", { "targetEps" }, 0.5f),
		_argTolerance(group, "Tolerance", "Tolerance", { "targetTolerance" }, 0.03f),		
		_argN(group, "N", "N particles", {'N','n'}, 8),
		_argSize(group, "Size", "Size of particle, accepts either one argument x or comma separated x,y,z.", { 's', "size" }, vec3(0.5f)),
		_argOutput(group, "Output", "Output .bin file.", {'o',"output"}, "out.bin"),
		_argMaxIter(group, "Max Iterations", "", {"maxIter"}, 2000),
		_argSchedule(group, "Schedule", "Schedule", { "schedule" }, 0.99f),
		_argRasterSize(group, "Raster Size", "Raster Size", {"raster"}, fast::ivec3(256)),
		_outputModule(group)
	{

	}

	virtual void prepare() override {
		_sdfpacking = std::make_unique<fast::SDFPacking>();
		
		_outputModule.prepare();
		/*fast::vec3 size = vec3(
			std::get<0>(_argSize.Get()),
			std::get<1>(_argSize.Get()),
			std::get<2>(_argSize.Get())
		);*/

		//fast::vec3 size = {_argSizeX.Get(), _argSizeY.Get(), _argSizeZ.Get() };
		

		
	}

	virtual void execute() override {
		if (!_sdfpacking)
			throw args::Error("SDFPacking not initialized");

		
		float bestAchieavablePorosity = 0.0f;
		const auto domain = fast::AABB(vec3(-1), vec3(1));
		const fast::vec3 size = _argSize.Get();
		const float v0 = size.x * size.y * size.z * (4.0f / 3.0f) * glm::pi<float>();
		const float v0rel = v0 / domain.volume();

		int N = _argN.Get();
		if (_argTargetEps) {
			N = (1.0f - _argTargetEps.Get()) / v0rel;		
		}		
		bestAchieavablePorosity = (1.0f - N * v0rel);

		/*if (N == 5)
			exit(77);*/

		//Add particles
		_sdfpacking->addEllipse(N, size);

		//Init
		fast::SDFPacking::Params p;
		p.maxIter = _argMaxIter.Get();
		p.annealingSchedule = _argSchedule.Get();
		_sdfpacking->init(p);

		//Iterate
		size_t iterCnt = 0;		

		std::set<float> scoreSet;

		while (_sdfpacking->step()) {	

			/*float diff = glm::abs(_sdfpacking->getSA().bestStateScore - bestAchieavablePorosity);
			float diffParticleRel = (diff * domain.volume()) / v0;*/

			if (_sdfpacking->getSA().bestStateScore < 0.0000003f)
				break;

			if (iterCnt % 1 == 0) {
				std::cout << "[" << iterCnt << "]\t";
				std::cout << _sdfpacking->getSA().bestStateScore << std::endl;
				//std::cout << "diff " << diff << " => " << diffParticleRel << " particles | tol: " << _argTolerance.Get() << std::endl;
				std::cout.flush();
				/*if (diffParticleRel < _argTolerance.Get())
					break;*/
			}

			if (_sdfpacking->getSA().rejections > 100) {
				std::cout << "Too many rejections. Stopping." << std::endl;
				break;
			}

			

			if(_sdfpacking->getSA().bestStateScore < _argTolerance.Get() && 
				scoreSet.find(_sdfpacking->getSA().bestStateScore) == scoreSet.end()
				)
			{
				std::cout << "Rasterizing\n";
				Volume v(_argRasterSize.Get(), TYPE_UCHAR);
				v.getPtr().createTexture();

				_sdfpacking->setRasterVolume(&v);
				_sdfpacking->rasterize(true);
				_sdfpacking->setRasterVolume(nullptr);

				std::cout << "Saving to " << _argOutput.Get() << std::endl;

				if (saveVolumeBinary(_argOutput.Get().c_str(), v)) {
					std::cout << "Complete" << std::endl;
				}
				else {
					std::cerr << "Failed to save volume" << std::endl;
				}

				float porosity = getPorosity<float>(v);

				//////////////////////////////////////
				RunRow gr;
				gr["px"] = _argSize.Get().x;
				gr["py"] = _argSize.Get().y;
				gr["pz"] = _argSize.Get().z;
				gr["pN"] = N;
				gr["porosityAchievable"] = bestAchieavablePorosity;
				gr["porosity"] = porosity;
				gr["overlap"] = _sdfpacking->getSA().bestStateScore;
				gr["schedule"] = _argSchedule.Get();
				gr["iterations"] = int(iterCnt);

				{
					/*std::cout << "rad basic.. " << std::endl;
					args::ArgumentParser parser("N/A", "N/A");
					AlphaModule m(parser);
					parser.ParseArgs(std::vector<std::string>{
						_argOutput.Get(),
							"-o", "alphaOverlap.csv",
							"-dall",
							"--basic"
					});
					m.prepare();
					m.getOutputModule().addGlobalRunRow(gr);
					m.execute();*/					
					float radBasic = fast::getReactiveAreaDensity<double>(v, v.dim(), 0.5f);
					gr["radBasic"] = radBasic;
					_outputModule.addRunRow(gr);
				}
				//////////////////////////////////////
			}

			scoreSet.insert(_sdfpacking->getSA().bestStateScore);

			iterCnt++;
		}		

		/*if (_sdfpacking->getSA().bestStateScore > _argTolerance.Get()) {
			exit(77);
		}*/
		

		
	}


private:

	CSVOutputModule _outputModule;

	
	args::ValueFlag<float> _argTargetEps; //Either minimize porosity
	//args::ValueFlag<float> _argTargetOverlap; //Or minimize overlap

	args::ValueFlag<float> _argTolerance; //Stop if reached within tolerance



	args::ValueFlag<int> _argMaxIter;
	args::ValueFlag<float> _argSchedule;
	args::ValueFlag<int> _argN;
	

	args::ValueFlag<fast::vec3, Vec3Reader> _argSize;

	std::unique_ptr<fast::SDFPacking> _sdfpacking;
	args::ValueFlag<fast::ivec3, IVec3Reader> _argRasterSize;

	args::ValueFlag<std::string> _argOutput;


};