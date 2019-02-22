#pragma once

#include <string>
#include <tuple>

//Directory, file
std::tuple<std::string, std::string> imguiFileExplorer(
	const std::string & directory,
	const std::string & extension,
	bool canDelete = false
);