#pragma once
#include "Module.h"
#include "VolumeIOModule.h"
#include <fastlib/volume/VolumeSurface.h>
#include <fastlib/volume/VolumeSegmentation.h>


class AlphaModule : public Module {

public:
	AlphaModule(args::Group & group) :
		_inputModule(group),
		_outputModule(group),
		_argDir(group, "string", "Direction (x|y|z)|all|pos|neg", { 'd', "dir" }, "x-"),
		_argBasic(group, "basic", "Basic calculation that doesn't care about unreachable areas", { "basic" }),
		_argExportMesh(group, "mesh", "Exports triangle mesh as Wavefront .OBJ file", { "mesh" }, ""),
		_argExportSegmentInfo(group, "segmentinfo", "Exports unaccessible segment information", { "segmentinfo" }, "")
	{

	}


	virtual void prepare() override {
		_inputModule.prepare();
		_outputModule.prepare();
	}

	static std::vector<RunRow> getAlpha(
		fast::Volume & volume,
		const std::string & name,
		bool basic,
		const std::vector<Dir> & dirs,
		const std::string & exportMesh = "",
		const std::string & exportInfo = ""
	){

		std::unique_ptr<CSVOutputModule> infoOutput;
		std::unique_ptr<args::ArgumentParser> tmpParser;
		if (exportInfo.length() > 0) {			
			tmpParser = std::make_unique<args::ArgumentParser>("tmp");
			infoOutput = std::make_unique<CSVOutputModule>(*tmpParser);
			tmpParser->ParseArgs(std::vector<std::string>{"-o",exportInfo});
		}

		std::vector<RunRow> res;
		RunRow o;
		o["ID"] = name;
		o["dimx"] = volume.dim().x;
		o["dimy"] = volume.dim().y;
		o["dimz"] = volume.dim().z;

		float porosity = getPorosity<float>(volume);
		o["porosity"] = porosity;

		if (!basic) {
			std::array<double, 6> radTensor;

			auto t0 = std::chrono::system_clock::now();
			auto ccl = fast::getVolumeCCL(volume, 255);
			radTensor = fast::getReactiveAreaDensityTensor<double>(ccl);
			auto t1 = std::chrono::system_clock::now();

			std::chrono::duration<double> dt = t1 - t0;
			double radTime = dt.count();



			std::vector<fast::CCLSegmentInfo> sinfo;
			if (exportInfo.length() > 0) {
				ccl.labels->getPtr().retrieve();
				sinfo = fast::getCCLSegmentInfo(ccl);
			}
			
			for (auto dir : dirs) {
				o["rad"] = radTensor[int(dir)];
				o["radTime"] = radTime / 6.0f;
				o["dir"] = dirString(dir);
				if (sinfo.size() > 0) {
					auto filtered = fast::filterSegmentInfo(sinfo, ccl, dir);
					float segmentRatio = fast::getCCLSegmentRatio(filtered, ccl);
					o["segmentVolumeRatio"] = segmentRatio;
				}

				res.push_back(o);
			}

			//Dir_none
			{
				auto t0 = std::chrono::system_clock::now();
				float radBasic = fast::getReactiveAreaDensity<double>(volume, volume.dim(), 0.5f);
				auto t1 = std::chrono::system_clock::now();
				std::chrono::duration<double> dt = t1 - t0;
				double radBasicTime = dt.count();

				o["rad"] = radBasic;
				o["radTime"] = radBasicTime;
				o["dir"] = dirString(DIR_NONE);

				if (sinfo.size() > 0) {
					auto filtered = fast::filterSegmentInfo(sinfo, ccl, DIR_NONE);
					float segmentRatio = fast::getCCLSegmentRatio(filtered, ccl);
					o["segmentVolumeRatio"] = segmentRatio;
				}

				res.push_back(o);
			}


			if (sinfo.size() > 0) {
				auto filtered = fast::filterSegmentInfo(sinfo, ccl, DIR_NONE);
				for (auto & si : filtered){
					RunRow r;
					r["ID"] = name;
					r["minX"] = si.minBB.x;
					r["minY"] = si.minBB.y;
					r["minZ"] = si.minBB.z;
					r["maxX"] = si.maxBB.x;
					r["maxY"] = si.maxBB.y;
					r["maxZ"] = si.maxBB.z;
					r["labelID"] = si.labelID;
					r["voxelNum"] = si.voxelNum;
					r["atBoundary"] = si.atBoundary;					
					infoOutput->addRunRow(r);
				}
							
			}




		}
		else {
			auto t0 = std::chrono::system_clock::now();
			float radBasic = fast::getReactiveAreaDensity<double>(volume, volume.dim(), 0.5f);
			auto t1 = std::chrono::system_clock::now();

			std::chrono::duration<double> dt = t1 - t0;
			double radBasicTime = dt.count();

			for (auto dir : dirs) {
				o["radBasic"] = radBasic;
				o["radBasicTime"] = radBasicTime;
				res.push_back(o);
			}
		}

		if (exportMesh.length() > 0) {
			

			
			//fast::Volume::enableOpenGLInterop = true;

			auto tmpVol = volume.withZeroPadding(ivec3(1), ivec3(1));		
			tmpVol.getPtr().createTexture();
			auto meshoutput = fast::getVolumeAreaMesh(tmpVol, false);


			//fast::getVolumeAreaMesh(tmpVol);

			if (meshoutput.Nverts > 0) {

				std::cout << "Saving " << exportMesh << ", Vertices:" << meshoutput.Nverts << std::endl;
				fast::CUDA_VBO::saveObj(meshoutput.data, exportMesh.c_str());
				std::cout << "Done." << std::endl;
				
			}
			else {
				std::cerr << "Has no triangles" << std::endl;
			}


		}


		

		return res;
	
	}

	virtual void execute() override {

		while (true) {
			auto volume = _inputModule.getNext();
			if (!volume) break;

			volume->getPtr().createTexture();

			auto rows = getAlpha(
				*volume,
				_inputModule.getVolumeName(volume.get()),
				_argBasic.Get(),
				getDirs(_argDir.Get()),
				_argExportMesh.Get(),
				_argExportSegmentInfo.Get()
			);

			for(auto row : rows)
				_outputModule.addRunRow(row);			
		}
	
	}

	CSVOutputModule & getOutputModule() {
		return _outputModule;
	}

private:

	VolumeInputModule _inputModule;
	CSVOutputModule _outputModule;

	args::ValueFlag<std::string> _argDir;
	args::Flag _argBasic;
	args::ValueFlag<std::string> _argExportMesh;
	args::ValueFlag<std::string> _argExportSegmentInfo;
	
	

};