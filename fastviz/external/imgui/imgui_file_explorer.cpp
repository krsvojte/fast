#include "imgui_file_explorer.h"
#include "imgui.h"


#include <fastlib/volume/VolumeIO.h>

#include <sstream>

using namespace std;



std::string filesizeString(uintmax_t size) {

	std::stringstream ss;

	if (size < 1024)
		ss << size << " B";
	else if(size < 1024*1024)
		ss << size/1024 << " kB";
	else if (size < 1024 * 1024 * 1024)
		ss << size / (1024*1024) << " MB";
	else if (size < uintmax_t(1024) * 1024 * 1024 * 1024)
		ss << size / (1024 * 1024 * 1024) << " GB";

	return ss.str();
}

tuple<string, string>  imguiFileExplorer(
	const string & directory, 
	const string & extension, 
	bool canDelete /*= false */)
{

	
	std::hash<std::string> hash_fn;
	int ID = static_cast<int>(hash_fn(directory));
	
	std::string path = (directory.length() == 0) ? fast::getCwd() : directory;
	

	tuple<string, string> result = std::make_tuple(path, "");// { path, "" };

	

	ImGui::PushID(ID);

	ImGui::BeginChildFrame(ID, ImVec2(ImGui::GetWindowContentRegionWidth(), 300));

	ImGui::Text(path.c_str());

	ImGui::Columns(1);	



	auto files = fast::listDir(path);
	for (auto & f : files) {

		std::string absPath = path + "/" + f;

		const auto & p = absPath;

		bool isDir = fast::isDir(absPath);

		
		if (!isDir &&  extension != "" && extension != ".*"){
			auto ext = fast::getExtension(p);
			if (!fast::checkExtension(f, extension)) continue;
			
		}
		

		if (ImGui::Selectable(f.c_str(), false)) {
			if (isDir) {
				std::get<0>(result) = p;				
			}
			else {
				std::get<1>(result) = p;
			}
		}

		/*ImGui::NextColumn();
		if (isDir) {
			ImGui::Text("DIR");
		}
		else {			
			ImGui::Text(filesizeString(fast::getFilesize(f)).c_str());
		}*/
		//..

		//ImGui::NextColumn();
		//..


		ImGui::NextColumn();	

	}

	ImGui::Columns(1);

	ImGui::EndChildFrame();
	ImGui::PopID();

	return result;


}
