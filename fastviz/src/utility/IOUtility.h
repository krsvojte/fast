#pragma once

#include <string>
#include <vector>
#include <ostream>
#include <sstream>

std::string getBaseDirectory(const std::string & filepath);

std::pair<std::string, std::string> getBaseAndFile(const std::string & filepath);

std::string readFileToString(const std::string & filepath);

std::string readFileWithIncludes(const std::string & filepath);

std::vector<std::string> split(const std::string & str, char delimiter);


void streamprintf(std::ostream & os, const char * format);

template<typename T, typename ... Types>
void streamprintf(std::ostream & os, const char * format, T value, Types ... args) {
	for (; *format != '\0'; format++) {
		if (*format == '%') {
			os << value;
			streamprintf(os, format + 1, args ...);
			return;
		}
		os << *format;
	}
}

template<typename... Types>
std::string toString(const char * format, Types ... args) {
	std::ostringstream ss;
	streamprintf(ss, format, args ...);
	return ss.str();	
}

std::string timestampString(const std::string & format = "%Y_%m_%d_%H_%M_%S");



