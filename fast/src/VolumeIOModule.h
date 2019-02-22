#pragma once
#include "Module.h"
#include "utility.h"

#include <fastlib/volume/VolumeIO.h>
#include <fastlib/geometry/GeometryIO.h>
#include <fastlib/volume/VolumeGenerator.h>
#include <fastlib/volume/VolumeRasterization.h>
#include <fastlib/utility/RandomGenerator.h>

#include <iostream>
#include <fstream>
#include <vector>

struct VolumeInput {
	std::string path;
	int index = 0;
};

std::vector<VolumeInput> getVolumeInputs(const std::string & path) {
	std::vector<VolumeInput> res;

	if (fast::checkExtension(path, "pos")) {
		std::cout << "Reading pos file ... ";
		std::ifstream f(path);
		int count = fast::getPosFileCount(f);
		std::cout << " found " << count << std::endl;

		for (int i = 0; i < count; i++) {
			res.push_back({ path, i });
		}
	}
	else {
		res.push_back({ path, 0 });
	}

	return res;

}



std::unique_ptr<fast::Volume> loadVolume(
	const VolumeInput & input,
	std::string & nameOut,
	fast::ivec3 resolution = fast::ivec3(0)
) {
	fast::Volume::enableOpenGLInterop = false;

	std::unique_ptr<fast::Volume> c;

	/*
	Load input volume
	*/	
	const std::string & filepath = input.path;
	

	if (fast::isDir(filepath)) {
		try {
			c = std::make_unique<fast::Volume>(
				fast::loadTiffFolder(filepath.c_str(), true)
				);
			c->binarize(1.0f);
			nameOut = filepath;			
		}
		catch (const char * ex) {
			std::cerr << "Failed to load: ";
			std::cerr << ex << std::endl;
			return nullptr;
		}
	}
	else if (fast::checkExtension(filepath, "bin")) {

		try {
			c = std::make_unique<fast::Volume>(
				fast::loadVolumeBinary(filepath.c_str())
				);
			nameOut = filepath;
		}
		catch (const char * ex) {
			std::cerr << "Failed to load: ";
			std::cerr << ex << std::endl;
			return nullptr;
		}
	}
	else if (fast::checkExtension(filepath, "sph")) {

		if (!(resolution.x >= 0 && resolution.y >= 0 && resolution.x >= 0))
			return nullptr;

		std::ifstream f(filepath);
		auto spheres = fast::loadSpheres(f);
		c = std::make_unique<fast::Volume>(
			fast::rasterizeSpheres(resolution, spheres)
			);
		nameOut = filepath;
		
	}
	else if (fast::checkExtension(filepath, "pos")) {
		
		std::cout << "Index " << input.index << std::endl;
		char buf[1024];
		sprintf(buf, "%s:%d", filepath.c_str(), input.index);
		nameOut = buf;

		
		std::ifstream f(filepath);
		const float scale = 1.0f / glm::pow(3.0f, 1.0f / 3.0f);
		const fast::AABB bb = { fast::vec3(0), fast::vec3(scale) };
		auto geometry = fast::readPosFile(f, input.index, bb);
		f.close();


		if (resolution.x == 0) {
			resolution = fast::ivec3(128);
		}		
		
		c = std::make_unique<fast::Volume>(resolution, TYPE_UCHAR);
		fast::rasterize(geometry, *c);

		std::cout << "Rasterized " << geometry.size() << " particles" << std::endl;
		std::cout << "Resolution: " << resolution.x << ", " << resolution.y << ", " << resolution.z << std::endl;
		
	}


	return std::move(c);

}





class VolumeInputModule : public Module {

public:

	VolumeInputModule(args::Group & parentGroup) :
		_argInput(parentGroup, "input", "Input file or folder", args::Options::Required),		
		_argSubvolume(parentGroup, "subvolume", "Sub Volume. Accepts either x or comma separated x,y,z.", { "sub" }, fast::ivec3(0)),
		_argOrigin(parentGroup, "origin", "Origin/Offset coordinates. Accepts either x or comma separated x,y,z.", { "origin" }, fast::ivec3(0)),
		_argRandomSubvolume(parentGroup, "randsub", "Random Sub Volume. Accepts either x or comma separated x,y,z.", { "randsub" }, fast::ivec3(0)),
		_argAspect(parentGroup, "aspect", "Aspect ratio crop. One dimension must be 1.0.", { "aspect" }, fast::vec3(1.0f))
		
	{

	}

	virtual void prepare() override {
		if (_argInput.Get().length() == 0) {
			throw args::Error("Invalid input");
		}

		_currentIndex = 0;
		_volumeInputs = getVolumeInputs(_argInput.Get());		
	}

	std::unique_ptr<fast::Volume> getNext(){
		if (_currentIndex >= _volumeInputs.size())
			return nullptr;
		
		fast::ivec3 loadResolution = _argSubvolume ? fast::ivec3(_argSubvolume.Get()) : fast::ivec3(0);
		
		std::string name;
		std::unique_ptr<fast::Volume> volume = loadVolume(_volumeInputs[_currentIndex], name, loadResolution);
		bool isPosFile = fast::checkExtension(_volumeInputs[_currentIndex].path, "pos");
		
		if (!volume)
			throw args::Error("Invalid volume");

		volume->getPtr().allocCPU();
		volume->getPtr().retrieve();

		//Resize if needed
		if ((_argSubvolume || _argOrigin != 0) && !isPosFile) {
			auto dim = volume->dim();

			fast::ivec3 minB = _argOrigin.Get();
			fast::ivec3 maxB = minB + _argSubvolume.Get();
			minB = glm::clamp(minB, fast::ivec3(0), dim);
			maxB = glm::clamp(maxB, fast::ivec3(0), dim);

			std::cerr << "Resizing: " << std::endl;
			std::cerr << minB.x << ", " << minB.y << ", " << minB.z << std::endl;
			std::cerr << maxB.x << ", " << maxB.y << ", " << maxB.z << std::endl;
			std::cerr << "dim:" << std::endl;
			std::cerr << dim.x << ", " << dim.y << ", " << dim.z << std::endl;

			if (maxB.x <= minB.x || maxB.y <= minB.y || maxB.z <= minB.z || maxB.x > dim.x || maxB.y > dim.y || maxB.z > dim.z) {
				std::cerr << "Random origin: " << std::endl;
				std::cerr << minB.x << ", " << minB.y << ", " << minB.z << std::endl;
				throw args::Error("Invalid resize options");
			}

			volume->resize(minB, maxB - minB);
		}
		else if (_argRandomSubvolume) {
			fast::RNGUniformInt rng(0, INT_MAX);
			rng.seedByTime();

			fast::ivec3 size = _argRandomSubvolume.Get();
			auto dim = volume->dim();
			fast::ivec3 maxOrigin = dim - size;
			fast::ivec3 minB = fast::ivec3(rng.next() % maxOrigin.x, rng.next() % maxOrigin.y, rng.next() % maxOrigin.z);
			fast::ivec3 maxB = minB + size;

			if (maxB.x <= minB.x || maxB.y <= minB.y || maxB.z <= minB.z || maxB.x > dim.x || maxB.y > dim.y || maxB.z > dim.z) {
				std::cerr << "Random origin: " << std::endl;
				std::cerr << minB.x << ", " << minB.y << ", " << minB.z << std::endl;
				throw args::Error("Invalid resize options");
			}			

			volume->resize(minB, fast::ivec3(size));
		}
		

		if (_argAspect) {
			auto aspect = _argAspect.Get();
			auto size = volume->dim();

			int k = -1;
			if (aspect.x == 1.0f)
				k = 0;
			else if (aspect.y == 1.0f)
				k = 1;
			else if (aspect.z == 1.0f)
				k = 2;
			

			if (k != -1) {

				fast::ivec3 newSize = {
					size[k] * aspect[0],
					size[k] * aspect[1],
					size[k] * aspect[2]
				};

				newSize = glm::clamp(newSize, fast::ivec3(0), size);
				volume->resize(fast::ivec3(0), fast::ivec3(newSize));
			}
			else {
				throw args::Error("Invalid aspect options, one dimension must equal to 1.0");
			}
		}



		_volumeNames[volume.get()] = name;
		_currentIndex++;

		return std::move(volume);		
	}

	std::string getVolumeName(const fast::Volume * volPtr) const{
		auto it = _volumeNames.find(volPtr);
		if (it == _volumeNames.end())
			throw args::Error("Invalid volume name");
		return it->second;
	}

	

protected:

	std::vector<VolumeInput> _volumeInputs;
	int _currentIndex;
	std::unordered_map<const fast::Volume *, std::string> _volumeNames;

	args::Positional<std::string> _argInput;
	
	args::ValueFlag<fast::ivec3, IVec3Reader> _argSubvolume;	
	args::ValueFlag<fast::ivec3, IVec3Reader> _argOrigin;
	args::ValueFlag<fast::ivec3, IVec3Reader> _argRandomSubvolume;

	args::ValueFlag<fast::vec3, Vec3Reader> _argAspect;

};


