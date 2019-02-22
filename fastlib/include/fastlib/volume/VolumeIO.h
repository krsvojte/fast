#pragma once

#include <fastlib/FastLibDef.h>
#include <fastlib/volume/Volume.h>

namespace fast {

	/*
		Throws const char * exception on failure
		TODO: nothrow
	*/	
	FAST_EXPORT Volume loadTiffFolder(const char * folder, bool commitToGPU = true);

	FAST_EXPORT bool saveVolumeBinary(const char * path, const Volume & channel);
	FAST_EXPORT Volume loadVolumeBinary(const char * path);

	FAST_EXPORT bool checkExtension(const std::string & path, const std::string & ext);

	FAST_EXPORT std::vector<std::string> listDir(const std::string & path);

	FAST_EXPORT bool isDir(const std::string & path);

	FAST_EXPORT std::string getCwd();


	FAST_EXPORT size_t getFilesize(const std::string & path);

	FAST_EXPORT std::string getExtension(const std::string & path);

	FAST_EXPORT bool saveSlicePNG(
		const char * path, 
		const Volume & vol,
		Dir dir,
		int index
	);




}