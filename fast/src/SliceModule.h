#pragma once
#include "Module.h"
#include "VolumeIOModule.h"

#include "AlphaModule.h"
#include "TauModule.h"
#include "utility.h"

class SliceModule : public Module {

public:
	SliceModule(args::Group & group) :
		_inputModule(group),
		_outputModule(group),
		_argSize(group, "size", "Size of subvolumes", { "size" }, fast::ivec3(32)),
		_argStride(group, "Stride", "Stride", { "stride" }, fast::ivec3(1)),
		_argBegin(group, "Begin", "Begin coord", { "begin" }, fast::ivec3(0)),
		_argEnd(group, "End", "End coord", { "end" }, fast::ivec3(0)),
		_argOutputFolder(group, "Output Folder", "", { "out" }, "out")
	{

	}


	virtual void prepare() override {
		_inputModule.prepare();
		
		
		_outputModule.overridePath(_argOutputFolder.Get() + "/data.csv");

		_outputModule.prepare();		
	}

	

	virtual void execute() override {

		const std::vector<Dir> dirs = getDirs("pos") ;

		using ivec3 = fast::ivec3;

		auto stride = _argStride.Get();
		auto size = _argSize.Get();

		const bool doAlpha = true;
		const bool doTau = true;

		double tolerance = double(pow(10.0, -5));

		double avgTime = 0.0;
		int avgTimeNum = 0;

		while (true) {
			auto volumePtr = _inputModule.getNext();			
			if (!volumePtr) break;
			auto name = _inputModule.getVolumeName(volumePtr.get());
			auto & volume = *volumePtr;

			name = utils::string_replace(name, "\\", "/");
			auto paths = explode(name,'/');
			auto stem = paths.back();



			auto begin = glm::min(_argBegin.Get(), volume.dim() - ivec3(1));
			auto end = glm::max(_argEnd.Get(), ivec3(0));
			if (end.x == 0) end.x = volume.dim().x - 1;
			if (end.y == 0) end.y = volume.dim().y - 1;
			if (end.z == 0) end.z = volume.dim().z - 1;


			int subCount = 0;
			

			std::string volDir = _argOutputFolder.Get() + "/" + stem;
//			utils::mkdir(volDir.c_str());

			
			int totalCount = 0;
			for (auto x0 = begin.x; x0 + size.x <= end.x; x0 += stride.x) {				
				for (auto y0 = begin.y; y0 + size.y <= end.y; y0 += stride.y) {					
					for (auto z0 = begin.z; z0 + size.z <= end.z; z0 += stride.z) {
						totalCount++;
					}
				}
			}			
			std::cout << "Subvolume count " << totalCount << std::endl;

			if (totalCount == 0) {
				std::cout << "Volume is not big enough " << volume.dim() << ", need " << size << std::endl;
				continue;
			}
			

			for (auto x0 = begin.x; x0 + size.x <= end.x; x0 += stride.x) {
				for (auto y0 = begin.y; y0 + size.y <= end.y; y0 += stride.y) {
					for (auto z0 = begin.z; z0 + size.z <= end.z; z0 += stride.z) {

						auto t0 = std::chrono::system_clock::now();

						ivec3 origin = { x0,y0,z0 };
						ivec3 dim = size;
						ivec3 outer = origin + size;
						std::cout << "Subvolume " << origin << " to " << outer;
						std::cout << " | " << subCount << "/" << totalCount;
						std::cout << " (" << (subCount / float(totalCount)) * 100 << "%)" << std::endl;

						size_t totalSize = size.x * size.y * size.z;

						auto subvolume = volume.getSubvolume(origin, size);

						
						char subvolstr[256];
						std::sprintf(subvolstr, "%04d_%04d_%04d.bin", origin.x, origin.y, origin.z);						
						std::string subvolPath = volDir + "/" + std::string(subvolstr);


						subvolume.getPtr().createTexture();


						RunRow rowAlpha;
						if(doAlpha)
							rowAlpha = AlphaModule::getAlpha(subvolume, name, true, { X_POS })[0];

						std::vector<RunRow> rowsTau;
						if (doTau) {
							int cpuThreshold = 72*72*72;
							rowsTau = TauModule::getTau(
								subvolume,
								name, (totalSize < cpuThreshold), -1, false,
								dirs, TYPE_FLOAT, DSOLVER_CG, 10000,
								tolerance, false
							);
						}

	
#ifdef PER_SLICE
						for (int k = 0; k < dirs.size(); k++) {
							Dir dir = dirs[k];

							std::string directionDir = subvolDir + "/" + dirString(dir);
							utils::mkdir(directionDir.c_str());

							
							auto dirIndex = getDirIndex(dir);
							RunRow o;
							o["IDVolume"] = name;
							if (doAlpha) {								
								o["alpha"] = rowAlpha["radBasic"];
								o["porosity"] = rowAlpha["porosity"];
							}

							if (doTau) {
								o["tau"] = rowsTau[k]["tau"];
							}
							
							o["sizex"] = size.x;
							o["sizey"] = size.y;
							o["sizez"] = size.z;
							o["originx"] = origin.x;
							o["originy"] = origin.y;
							o["originz"] = origin.z;
							o["dir"] = dirString(dir);
							

							char buf[256];
							for (auto sid = 0; sid < subvolume.dim()[dirIndex]; sid++) {
								std::sprintf(buf,"%04d",sid);
								
								//stem + "_" + dirString(dir) + "_slice_" + std::string(buf)
								std::string filename = std::string(buf)
									+ ".png";
								std::string path = directionDir + "/" + filename;

								RunRow oo = o;
								size_t sublen = _argOutputFolder.Get().length() + 1;
								oo["ID"] = std::string(path,sublen, path.length() - sublen);

								saveSlicePNG(
									path.c_str(), 
									subvolume, 
									dir, 
									sid
								);
								
								_outputModule.addRunRow(oo);
							}
						

							//Slicing
							
						}			
#else

						utils::mkdir(volDir.c_str());						
						fast::saveVolumeBinary(subvolPath.c_str(), subvolume);

						for (int k = 0; k < dirs.size(); k++) {

							const Dir dir = dirs[k];
							const auto dirIndex = getDirIndex(dir);
							RunRow o;

							size_t sublen = _argOutputFolder.Get().length() + 1;
							o["ID"] = std::string(subvolPath, sublen, subvolPath.length() - sublen);

							//o["IDVolume"] = name;
							if (doAlpha) {
								o["alpha"] = rowAlpha["radBasic"];
								o["porosity"] = rowAlpha["porosity"];
							}

							if (doTau) {
								o["tau"] = rowsTau[k]["tau"];
							}

							o["sizex"] = size.x;
							o["sizey"] = size.y;
							o["sizez"] = size.z;
							o["originx"] = origin.x;
							o["originy"] = origin.y;
							o["originz"] = origin.z;
							o["dir"] = dirString(dir);


							_outputModule.addRunRow(o);
						}

#endif
						

						//auto rowTau= AlphaModule::getAlpha(subvolume, name, true, { X_POS })[0];

						auto t1 = std::chrono::system_clock::now();
						std::chrono::duration<double> dt = t1 - t0;
						double subTime = dt.count();
						avgTime += subTime;
						avgTimeNum++;

						float ETAmin = ((avgTime / avgTimeNum) * (totalCount - subCount)) / 60.0f;

						std::cout << "time " << subTime << " | avg: " << avgTime / avgTimeNum << "s" << " | ETA: " << ETAmin << "m" << std::endl;


						subCount++;
					}
				}			
			}

			std::cout << "Generated " << subCount << " subvolumes" << std::endl;		
			
/*

			volume->getPtr().createTexture();

			auto rows = AlphaModule::getAlpha(
				*volume,
				_inputModule.getVolumeName(volume.get()),
				true,
				dirs
			);


			for (auto row : rows)
				_outputModule.addRunRow(row);*/
		}

	}

	CSVOutputModule & getOutputModule() {
		return _outputModule;
	}

private:

	VolumeInputModule _inputModule;
	CSVOutputModule _outputModule;

	

	args::ValueFlag<fast::ivec3, IVec3Reader> _argSize;
	args::ValueFlag<fast::ivec3, IVec3Reader> _argStride;
	args::ValueFlag<fast::ivec3, IVec3Reader> _argBegin;
	args::ValueFlag<fast::ivec3, IVec3Reader> _argEnd;

	args::ValueFlag<std::string> _argOutputFolder;
	

};