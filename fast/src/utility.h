#pragma once

#include <chrono>
#include <numeric>
#include <iomanip>

#include <fastlib/volume/VolumeMeasures.h>
#include <fastlib/volume/VolumeIO.h>
#include <fastlib/utility/PrimitiveTypes.h>



#include <fstream>

using namespace fast;


std::ostream& operator<<(std::ostream& os, fast::ivec3 & v)
{
	os << v.x << ',' << v.y << ',' << v.z;
	return os;
}

std::ostream& operator<<(std::ostream& os, fast::vec3 & v)
{
	os << v.x << ',' << v.y << ',' << v.z;
	return os;
}

struct Vec3Reader
{
	void operator()(const std::string &name, const std::string & value, fast::vec3 & v)
	{
		size_t commapos = 0;
		v[0] = std::stof(value, &commapos);
		if (commapos == value.length()) {
			v[1] = v[0];
			v[2] = v[0];
			return;
		}		
		size_t commapos2 = 0;
		v[1] = std::stof(std::string(value, commapos + 1), &commapos2);		
		v[2] = std::stof(std::string(value, commapos + commapos2 + 2));
	}
};

struct IVec3Reader
{
	void operator()(const std::string &name, const std::string & value, fast::ivec3 & v)
	{
		size_t commapos = 0;
		v[0] = std::stoi(value, &commapos);
		if (commapos == value.length()) {
			v[1] = v[0];
			v[2] = v[0];
			return;
		}
		size_t commapos2 = 0;
		v[1] = std::stoi(std::string(value, commapos + 1), &commapos2);
		v[2] = std::stoi(std::string(value, commapos + commapos2 + 2));
	}
};



inline std::string tmpstamp(const std::string & format /*= "%Y_%m_%d_%H_%M_%S"*/)
{
	char buffer[256];
	std::time_t now = std::time(NULL);
	std::tm * ptm = std::localtime(&now);
	std::strftime(buffer, 256, format.c_str(), ptm);
	return std::string(buffer);
}



/*
Extract what directions should be calculated from flagStr
all - positive and negative
pos, neg - three directions
x and/or y and/or z
default: x positive
*/
inline std::vector<Dir> getDirs(const std::string & flagStr) {

	if (flagStr == "all")
		return { X_POS, X_NEG,Y_POS,Y_NEG,Z_POS,Z_NEG };
	if (flagStr == "pos")
		return { X_POS, Y_POS, Z_POS };
	if (flagStr == "neg")
		return { X_NEG, Y_NEG, Z_NEG };

	if (flagStr == "xneg")
		return { X_NEG };
	if (flagStr == "yneg")
		return { Y_NEG };
	if (flagStr == "zneg")
		return { Z_NEG };


	std::vector<Dir> arr;
	for (auto & c : flagStr) {
		if (c == 'x') arr.push_back(X_POS);
		else if (c == 'y') arr.push_back(Y_POS);
		else if (c == 'z') arr.push_back(Z_POS);
	}


	if (arr.empty())
		return { X_POS };

	return arr;
}


inline std::string dirString(Dir d) {
	switch (d) {
	case X_POS: return "X_POS";
	case X_NEG: return "X_NEG";
	case Y_POS: return "Y_POS";
	case Y_NEG: return "Y_NEG";
	case Z_POS: return "Z_POS";
	case Z_NEG: return "Z_NEG";
	case DIR_NONE: return "DIR_NONE";
	}

	return "Undefined direction";
}

inline std::string solverString(fast::DiffusionSolverType type) {
	switch (type)
	{
	case fast::DSOLVER_BICGSTAB:
		return "BICGSTAB";
		break;	
	case fast::DSOLVER_CG:
		return "CG";
		break;	
	}
	return "Unknown solver";
}

//https://stackoverflow.com/questions/12966957/is-there-an-equivalent-in-c-of-phps-explode-function
inline std::vector<std::string> explode(std::string const & s, char delim)
{
	std::vector<std::string> result;
	std::istringstream iss(s);

	for (std::string token; std::getline(iss, token, delim); )
	{
		result.push_back(std::move(token));
	}

	return result;
}


#include <sstream>
#include <sys/stat.h>

// for windows mkdir
#ifdef _WIN32
#include <direct.h>
#endif

namespace utils
{
	/**
	* Checks if a folder exists
	* @param foldername path to the folder to check.
	* @return true if the folder exists, false otherwise.
	*/
	inline bool folder_exists(std::string foldername)
	{
		struct stat st;
		stat(foldername.c_str(), &st);
		return st.st_mode & S_IFDIR;
	}

	/**
	* Portable wrapper for mkdir. Internally used by mkdir()
	* @param[in] path the full path of the directory to create.
	* @return zero on success, otherwise -1.
	*/
	inline int _mkdir(const char *path)
	{
#ifdef _WIN32
		return ::_mkdir(path);
#else
#if _POSIX_C_SOURCE
		return ::mkdir(path);
#else
		return ::mkdir(path, 0755); // not sure if this works on mac
#endif
#endif
	}

	/**
	* Recursive, portable wrapper for mkdir.
	* @param[in] path the full path of the directory to create.
	* @return zero on success, otherwise -1.
	*/
	inline int mkdir(const char *path)
	{
		std::string current_level = "";
		std::string level;
		std::stringstream ss(path);

		// split path using slash as a separator
		while (std::getline(ss, level, '/'))
		{
			current_level += level; // append folder to the current level

									// create current level
			if (!folder_exists(current_level) && _mkdir(current_level.c_str()) != 0)
				return -1;

			current_level += "/"; // don't forget to append a slash
		}

		return 0;
	}

	std::string string_replace(const std::string& str, const std::string& match,
		const std::string& replacement, unsigned int max_replacements = UINT_MAX)
	{
		size_t pos = 0;
		std::string newstr = str;
		unsigned int replacements = 0;
		while ((pos = newstr.find(match, pos)) != std::string::npos
			&& replacements < max_replacements)
		{
			newstr.replace(pos, match.length(), replacement);
			pos += replacement.length();
			replacements++;
		}
		return newstr;
	}
}