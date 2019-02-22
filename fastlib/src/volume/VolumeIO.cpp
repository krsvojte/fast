#include "volume/VolumeIO.h"


#include "TinyTIFF/tinytiffreader.h"

#include "tinydir.h"
#ifdef _MSC_VER
#undef min
#undef max
#endif

#ifdef _MSC_VER
#include <direct.h>  
#include <stdlib.h>  
#include <stdio.h> 
#else
#include <unistd.h>
#endif

#include <fstream>
#include <iostream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

using namespace std;
using namespace fast;



bool tiffSize(const char * path, int *x, int *y, int * bytes, int *frames = nullptr)
{

	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(path);

	if (!tiffr) return false;

	uint32_t width = TinyTIFFReader_getWidth(tiffr);
	uint32_t height = TinyTIFFReader_getHeight(tiffr);

	*x = width;
	*y = height;

	*bytes = TinyTIFFReader_getSampleFormat(tiffr);

	//Count frames
	if (frames != nullptr) {
		int cnt = 0;
		do { cnt++; } while (TinyTIFFReader_readNext(tiffr));

		*frames = cnt;
	}

	return true;
}

bool readTiff(const char * path, void * buffer)
{
	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(path);
	if (!tiffr) return false;

	uint32_t width = TinyTIFFReader_getWidth(tiffr);
	uint32_t height = TinyTIFFReader_getHeight(tiffr);

	TinyTIFFReader_getSampleData(tiffr, buffer, 0);
	if (TinyTIFFReader_wasError(tiffr)) {
		std::cerr << "TinyTIFFReader Error: " << TinyTIFFReader_getLastError(tiffr) << std::endl;
		return false;
	}

	TinyTIFFReader_close(tiffr);
	return true;
}

FAST_EXPORT std::vector<std::string> fast::listDir(const std::string & path)
{

	std::vector<std::string> files;

	tinydir_dir dir;

	//Cant open dir
	if (tinydir_open(&dir, path.c_str()) == -1)
	{
		tinydir_close(&dir);
		return files;
	}

	while (dir.has_next)
	{
		tinydir_file file;
		if (tinydir_readfile(&dir, &file) == -1)
			break;

		files.push_back(file.name);
		
		if (tinydir_next(&dir) == -1)
			break;

	}

	tinydir_close(&dir);

	std::sort(files.begin(), files.end());

	return files;
}


FAST_EXPORT bool fast::isDir(const std::string & path)
{

	tinydir_dir d;
	return tinydir_open(&d, path.c_str()) != -1;
	
	/*if (tinydir_file_open(&f, path.c_str()) == -1)			
		return false;*/
	
	//return  f.is_dir;

}

FAST_EXPORT std::string fast::getCwd()
{
	
	
	char buf[512];
	char *cwd_result = getcwd(buf, 512);
	return std::string(buf);

}

FAST_EXPORT size_t fast::getFilesize(const std::string & path)
{
	if (isDir(path)) return 0;
	std::ifstream in(path, std::ifstream::ate | std::ifstream::binary);
	return in.tellg();
}

FAST_EXPORT std::string fast::getExtension(const std::string & path)
{
	tinydir_file file;
	if (tinydir_file_open(&file, path.c_str()) == -1)
		return "";
	return std::string(file.extension);
}





size_t directoryFileCount(const std::string & path, const std::string & ext)
{

	tinydir_dir dir;

	//Cant open dir
	if (tinydir_open(&dir, path.c_str()) == -1)
	{		
		tinydir_close(&dir);
		return 0;
	}

	size_t cnt = 0;


	while (dir.has_next)
	{
		tinydir_file file;
		if (tinydir_readfile(&dir, &file) == -1)				
			break;
		

		if (checkExtension(file.name, ext))
			cnt++;

		
		if (tinydir_next(&dir) == -1)		
			break;
		
	}

	tinydir_close(&dir);

	return cnt;

	
	//static_cast<bool(*)(const fs::path&)>(fs::is_regular_file));
}



Volume fast::loadTiffFolder(const char * folder, bool commitToGPU)
{
		

	//fs::path path(folder);

	if (!isDir(folder))
		throw "Volume directory not found";


	int numSlices = static_cast<int>(directoryFileCount(folder, "tiff"))
					+ static_cast<int>(directoryFileCount(folder, "tif"));

	int x, y, bytes;

	//Find first tiff
	for (auto & f : listDir(folder)) {
		if (isDir(f)) continue;		
		if (!checkExtension(f, "tiff") && !checkExtension(f, "tif")) continue;
		//if (f.path().extension() != ".tiff" && f.path().extension() != ".tif") continue;
		

		if (!tiffSize((std::string(folder) + "/" + f).c_str(), &x, &y, &bytes))
			throw "Couldn't read tiff file";
		else
			break;
	}
	

	if (bytes != 1)
		throw "only uint8 supported right now";
	

	Volume volume({ x,y,numSlices }, TYPE_UCHAR) ;

	uchar * ptr = (uchar*)volume.getPtr().getCPU();

	

	
	uint sliceIndex = 0;
	for (auto & f : listDir(folder)) {		
		if (isDir(f)) continue;
		if (!checkExtension(f, "tiff") && !checkExtension(f, "tif")) continue;
		
		
		if (!readTiff((std::string(folder) + "/" + f).c_str(), ptr + (sliceIndex * x * y))) {
			throw "Failed to read slices";
		}

		sliceIndex++;
	}	

	if(commitToGPU)
		volume.getPtr().commit();

	return volume;
}

FAST_EXPORT bool fast::saveVolumeBinary(const char * path, const Volume & channel)
{

	std::ofstream f(path, std::ios::binary);
	if (!f.good()) return false;

	const auto & dataptr = channel.getPtr();
	const void * data = dataptr.getCPU();

	//bool doubleBuffered = channel.isDoubleBuffered();
	bool deprecated = false;

	PrimitiveType type = channel.type();
	ivec3 dim = channel.dim();

	f.write((const char *)&type, sizeof(PrimitiveType));
	f.write((const char *)&dim, sizeof(ivec3));
	f.write((const char *)&deprecated, sizeof(bool));
	f.write((const char *)data, dataptr.byteSize());
	f.close();

	return true;
}

FAST_EXPORT Volume fast::loadVolumeBinary(const char * path)
{

	std::ifstream f(path, std::ios::binary);
	if (!f.good()) 
		throw "Couldn't read file";

	PrimitiveType type;
	ivec3 dim;
	bool deprecated;

	f.read((char *)&type, sizeof(PrimitiveType));
	f.read((char *)&dim, sizeof(ivec3));
	f.read((char *)&deprecated, sizeof(bool));


	Volume vol(dim, type);
	auto & dataptr = vol.getPtr();
	void * data = dataptr.getCPU();
	
	f.read((char *)data, dataptr.byteSize());

	dataptr.commit();

	return vol;
}

FAST_EXPORT bool fast::checkExtension(const std::string & path, const std::string & ext)
{
	return (path.substr(path.find_last_of(".") + 1) == ext);
}


FAST_EXPORT bool fast::saveSlicePNG(const char * path, const Volume & vol, Dir dir, int index)
{

	assert(vol.type() == TYPE_UCHAR);


	ivec2 dim = vol.getSliceDim(dir);
	size_t elemBytes = primitiveSizeof(vol.type());
	size_t sliceByteSize = dim.x*dim.y * elemBytes;

	std::vector<uchar> data(sliceByteSize, 0);

	vol.getSlice(dir, index, data.data());


	//const uchar * data = reinterpret_cast <const uchar*>(vol.getPtr().getCPU());
	//auto dim = vol.dim();
	//ivec3 stride = 
	int res = stbi_write_png(path, dim.x, dim.y, 1, data.data(), dim.x * elemBytes);

	return res != 0;
}