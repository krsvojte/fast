#include "Ui.h"

#include "BatteryApp.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"
#include "imgui/imgui_file_explorer.h"

#include <glm/gtc/type_ptr.hpp>

#include <fastlib/volume/VolumeIO.h>
#include <fastlib/volume/VolumeMeasures.h>
#include <fastlib/volume/VolumeGenerator.h>
#include <fastlib/volume/VolumeSegmentation.h>
#include <fastlib/volume/VolumeSurface.h>
#include <fastlib/utility/Timer.h>
#include <fastlib/optimization/SDFPacking.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include "utility/IOUtility.h"


void mayaStyle() {
	ImGuiStyle& style = ImGui::GetStyle();

	style.ChildWindowRounding = 3.f;
	style.GrabRounding = 0.f;
	style.WindowRounding = 0.f;
	style.ScrollbarRounding = 3.f;
	style.FrameRounding = 3.f;
	style.WindowTitleAlign = ImVec2(0.5f, 0.5f);

	style.Colors[ImGuiCol_Text] = ImVec4(0.73f, 0.73f, 0.73f, 1.00f);
	style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	style.Colors[ImGuiCol_WindowBg] = ImVec4(0.26f, 0.26f, 0.26f, 0.95f);
	style.Colors[ImGuiCol_ChildWindowBg] = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
	style.Colors[ImGuiCol_PopupBg] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_Border] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	style.Colors[ImGuiCol_TitleBg] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.21f, 0.21f, 0.21f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ComboBg] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
	style.Colors[ImGuiCol_CheckMark] = ImVec4(0.78f, 0.78f, 0.78f, 1.00f);
	style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.74f, 0.74f, 0.74f, 1.00f);
	style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.74f, 0.74f, 0.74f, 1.00f);
	style.Colors[ImGuiCol_Button] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.43f, 0.43f, 0.43f, 1.00f);
	style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
	style.Colors[ImGuiCol_Header] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_Column] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	style.Colors[ImGuiCol_ColumnHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ColumnActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_CloseButton] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
	style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.32f, 0.52f, 0.65f, 1.00f);
	style.Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.50f);
}


struct ImguiInputs {



	template <typename T>
	bool operator() (T & value) {
		ImGui::Text("Type Not implemented");
		return false;
	}

	bool operator() (float & value) {
		return ImGui::InputFloat("##value", &value, value / 100.0f);
	}

	bool operator() (int & value) {
		return ImGui::InputInt("##value", &value, value / 100);
	}

	bool operator() (bool & value) {
		bool res = ImGui::Checkbox("##value", &value);

		if (res) {
			char br;
			br = 0;
		}
		return res;
	}

	bool operator() (double & value) {
		float fv = static_cast<float>(value);
		bool changed = (*this)(fv);
		if (changed) {
			value = static_cast<double>(fv);
			return true;
		}
		return false;
	}

	bool operator() (std::string & value) {
		value.reserve(512);
		if (value.length() > 512)
			value.resize(512);
		return ImGui::InputTextMultiline("##value", &value[0], 512);
	}

	bool operator() (vec2 & value) {
		return ImGui::DragFloat2("##value", glm::value_ptr(value), 0.1f);
	}
	bool operator() (vec3 & value) {
		return ImGui::DragFloat3("##value", glm::value_ptr(value), 0.1f);
	}
	bool operator() (vec4 & value) {
		return ImGui::DragFloat4("##value", glm::value_ptr(value), 0.1f);
	}

	bool operator() (ivec2 & value) {
		return ImGui::DragInt2("##value", glm::value_ptr(value), 1);
	}
	bool operator() (ivec3 & value) {
		return ImGui::DragInt3("##value", glm::value_ptr(value), 1);
	}
	bool operator() (ivec4 & value) {
		return ImGui::DragInt4("##value", glm::value_ptr(value), 1);
	}

	bool operator() (mat2 & value) {
		return renderMatrix(glm::value_ptr(value), 2);
	}

	bool operator() (mat3 & value) {
		return renderMatrix(glm::value_ptr(value), 3);
	}

	bool operator() (mat4 & value) {
		return renderMatrix(glm::value_ptr(value), 4);
	}

private:

	bool renderMatrix(float * M, int dimension) {
		int ID = static_cast<int>(reinterpret_cast<size_t>(M));
		ImGui::BeginChildFrame(ID, ImVec2(ImGui::GetColumnWidth() - 17, ImGui::GetItemsLineHeightWithSpacing() * dimension + 5));
		ImGui::Columns(dimension);

		bool changed = false;
		for (int k = 0; k < dimension; k++) {
			for (int i = 0; i < dimension; i++) {
				ImGui::PushID(k*dimension + i);
				changed |= ImGui::DragFloat("##value", M + k + i*dimension);
				ImGui::PopID();
			}
			if (k < dimension - 1)
				ImGui::NextColumn();
		}

		ImGui::Columns(1);
		ImGui::EndChildFrame();
		return changed;
	}
};

bool renderOptionSet(const std::string & name, OptionSet & options, unsigned int depth)
{

	bool hasChanged = false;

	if (depth == 0) {
		ImGui::Columns(2);
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
	}

	const void * id = &options;
	ImGui::PushID(id);
	ImGui::AlignFirstTextHeightToWidgets();

	bool isOpen = (depth > 0) ? ImGui::TreeNode(id, "%s", name.c_str()) : true;


	ImGui::NextColumn();
	ImGui::AlignFirstTextHeightToWidgets();

	ImGui::NextColumn();

	if (isOpen) {

		for (auto & it : options.children) {
			hasChanged |= renderOptionSet(it.first, *it.second, depth + 1);
		}


		for (auto & opt : options.options) {
			const void * oid = &opt;
			ImGui::PushID(oid);

			ImGui::Bullet();
			ImGui::Selectable(opt.first.c_str());

			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::AlignFirstTextHeightToWidgets();

			hasChanged |= mpark::visit(ImguiInputs(), opt.second.value);

			ImGui::PopItemWidth();
			ImGui::NextColumn();



			ImGui::PopID();
		}


		if (depth > 0)
			ImGui::TreePop();

	}

	ImGui::PopID();


	if (depth == 0) {
		ImGui::Columns(1);
		ImGui::PopStyleVar();
	}

	return hasChanged;

}


Ui::Ui(BatteryApp & app) : _app(app)
{
	//gui
	ImGui_ImplGlfwGL3_Init(_app._window.handle, false);
	//mayaStyle();
}

void Ui::update(double dt)
{
	ImGui_ImplGlfwGL3_NewFrame();


	int w = static_cast<int>(_app._window.width * 0.35f);
	ImGui::SetNextWindowPos(
		ImVec2(static_cast<float>(_app._window.width - w), 0), 
		ImGuiSetCond_Always
	);

	if (_app._options["Render"].get<bool>("slices")) {
		ImGui::SetNextWindowSize(
			ImVec2(static_cast<float>(w), 2.0f * (static_cast<float>(_app._window.height) / 3.0f)),
			ImGuiSetCond_Always
		);
	}
	else {
		ImGui::SetNextWindowSize(
			ImVec2(static_cast<float>(w), static_cast<float>(_app._window.height)),
			ImGuiSetCond_Always
		);
	}

	static bool mainOpen = false;

	
	

	ImGui::Begin("Main", &mainOpen);

	//FPS display
	{
		double fps = _app.getFPS();
		ImVec4 color;
		if (fps < 30.0)
			color = ImVec4(1, 0, 0,1);
		else if (fps < 60.0)
			color = ImVec4(0, 0.5, 1,1);
		else
			color = ImVec4(0, 1, 0, 1);

		ImGui::TextColored(color, "FPS: %f", float(fps));
	}


	/*
		Options
	*/
	ImGui::Separator();
	ImGui::Text("Options");
	ImGui::Separator();

	if (ImGui::Button("Save")) {
		std::ofstream optFile(OPTIONS_FILENAME);		
			optFile << _app._options;		
	}
	ImGui::SameLine();
	if (ImGui::Button("Load")) {
		std::ifstream optFile(OPTIONS_FILENAME);
		optFile >> _app._options;
	}

	renderOptionSet("Options", _app._options, 0);

	
	ImGui::Separator();
	ImGui::Text("View");
	ImGui::Separator();

	ImGui::SliderFloat3("Slice (Min)", reinterpret_cast<float*>(&_app._options["Render"].get<vec3>("sliceMin")), -1, 1);
	ImGui::SliderFloat3("Slice (Max)", reinterpret_cast<float*>(&_app._options["Render"].get<vec3>("sliceMax")), -1, 1);


	{
		_app._volumeRaycaster->sliceMin = _app._options["Render"].get<vec3>("sliceMin");
		_app._volumeRaycaster->sliceMax = _app._options["Render"].get<vec3>("sliceMax");

		_app._volumeRaycaster->opacityWhite = _app._options["Render"].get<float>("opacityWhite");
		_app._volumeRaycaster->opacityBlack = _app._options["Render"].get<float>("opacityBlack");

		_app._volumeRaycaster->preserveAspectRatio = _app._options["Render"].get<bool>("preserveAspectRatio");

		_app._volumeRaycaster->showGradient = _app._options["Render"].get<bool>("showGradient");


		_app._volumeRaycaster->setNormalizeRange(
		{ 
			_app._options["Render"].get<float>("normalizeLow"), 
			_app._options["Render"].get<float>("normalizeHigh")
		}
		);		
	}	

	

	

	ImGui::SliderInt("RenderChannel", 
		&_app._options["Render"].get<int>("channel"),
		0, int(_app._volumes.size() - 1)
	);

	
	if (_app._volumes.size() > 0) {
		int channelID = _app._options["Render"].get<int>("channel");
		if (channelID >= int(_app._volumes.size())) {
			channelID = int(_app._volumes.size()) - 1;
			_app._options["Render"].set<int>("channel", channelID);
		}		
		auto it = std::next(_app._volumes.begin(), channelID);
		auto & renderChannel = *(it->second);

		ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s",
			it->first.c_str()			
		);
		ImGui::TextColored(ImVec4(1, 1, 0, 1), "%d %d %d",
			renderChannel.dim().x, renderChannel.dim().y, renderChannel.dim().z
		);

		ImGui::SameLine(); 

		if(ImGui::Button("Clear Channel"))
			renderChannel.clear();

		ImGui::SameLine();
		if (ImGui::Button("Normalize range")) {
			
			uchar buf[64];
			renderChannel.min(buf);
			float minVal = primitiveToNormFloat(renderChannel.type(), buf);
			renderChannel.max(buf);
			float maxVal = primitiveToNormFloat(renderChannel.type(), buf);

			std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;
						
			_app._options["Render"].set<float>("normalizeLow", float(minVal));
			_app._options["Render"].set<float>("normalizeHigh",float(maxVal));

		}

		if (ImGui::Button("Delete Channel")) {
			_app._volumes.erase(std::next(_app._volumes.begin(), channelID)->first);
			//_app._volume->removeChannel(channelID);
			_app._options["Render"].get<int>("channel") = (channelID) % int(_app._volumes.size());
		}

	}
	else {
		ImGui::TextColored(ImVec4(1, 0, 0, 1), "No Volumes Loaded");
	}


	/*
	Volume
	*/

	
	
	
	ImGui::Separator();
	ImGui::Text("Measurements");
	ImGui::Separator();

	
	/*static bool enableBiCGCPU = false;
	static bool enablefBiCGCPU = false;*/
	/*static bool enableBiCG = false;	
	static bool enablefBiCG = false;*/
	//static bool enableNUBiCG = true;
	//static bool enableFNUBiCG = false;
	/*static bool enableCG = false;
	static bool enablefCG = false;*/
	static bool tauVerbose = false;

	static bool singlePrec = true;
	static bool onDevice = true;
	static bool enableCG = true;
	static bool enableBICGSTAB = true;
	static int cpuThreads = -1;
	
	if (ImGui::Button("Tortuosity")) {
		

		/*
		Prepare
		*/
		fast::TortuosityParams tp;
		tp.coeffs = {
			_app._options["Diffusion"].get<float>("D_zero"),
			_app._options["Diffusion"].get<float>("D_one")
		};
		tp.dir = Dir(_app._options["Diffusion"].get<int>("direction"));
		tp.tolerance = powf(10.0, -_app._options["Diffusion"].get<int>("Tolerance"));
		tp.maxIter = size_t(_app._options["Diffusion"].get<int>("maxIter"));
	
		tp.coeffs[1] = powf(10.0, -_app._options["Diffusion"].get<int>("expD_one"));
		tp.verbose = tauVerbose;
		tp.onDevice = onDevice;
		tp.verbose = tauVerbose;
		tp.cpuThreads = cpuThreads;
		
		auto & mask = *_app._volumes[CHANNEL_MASK];					

		fast::Timer tt(true);

		//Optional output
		fast::Volume * outputChannel = nullptr;
		if (_app._volumes.find(CHANNEL_CONCETRATION) != _app._volumes.end()) {
			outputChannel = _app._volumes[CHANNEL_CONCETRATION].get();
		}	
		
		if (enableBICGSTAB) {
			fast::Timer bicgT(true);
			double tau = 0.0;
			if(singlePrec)
				tau = fast::getTortuosity<float>(mask, tp, fast::DiffusionSolverType::DSOLVER_BICGSTAB, outputChannel);
			else
				tau = fast::getTortuosity<double>(mask, tp, fast::DiffusionSolverType::DSOLVER_BICGSTAB, outputChannel);

			std::cout << "BICGSTAB\t\t" << tau << " (" << bicgT.time() << ")" << std::endl;
		}

		if (enableCG) {
			fast::Timer bicgT(true);
			double tau = 0.0;
			if (singlePrec)
				tau = fast::getTortuosity<float>(mask, tp, fast::DiffusionSolverType::DSOLVER_CG, outputChannel);
			else
				tau = fast::getTortuosity<double>(mask, tp, fast::DiffusionSolverType::DSOLVER_CG, outputChannel);

			std::cout << "CG\t\t" << tau << " (" << bicgT.time() << ")" << std::endl;
		}

		//std::cout << "Tau time: " << tt.time() << std::endl;
	}

	
	ImGui::Checkbox("CG", &enableCG); ImGui::SameLine();
	ImGui::Checkbox("BICG", &enableBICGSTAB);

	ImGui::Checkbox("Single", &singlePrec); ImGui::SameLine();
	ImGui::Checkbox("OnDevice", &onDevice);
	ImGui::Checkbox("Verbose", &tauVerbose);

	ImGui::InputInt("CpuThreads", &cpuThreads, 1, 8);
	
	static bool enableRADMesh = true;
	
	if (ImGui::Button("Pad by 1")) {
		_app._volumes[CHANNEL_MASK] = std::make_unique<fast::Volume>(
			_app._volumes[CHANNEL_MASK]->withZeroPadding(ivec3(1), ivec3(1))
			);
	}

	if (ImGui::Button("Porosity")) {
		std::cout << fast::getPorosity<double>(*_app._volumes[CHANNEL_MASK]) << std::endl;
	}
	
	if (ImGui::Button("Reactive Area Density")) {
		auto ccl = fast::getVolumeCCL(*_app._volumes[CHANNEL_MASK], 255);


		if (enableRADMesh) {
			Dir dir = Dir(_app._options["Diffusion"].get<int>("direction"));
			fast::Volume boundaryVolume = fast::generateBoundaryConnectedVolume(ccl, dir);
			boundaryVolume.getPtr().createTexture();

			//fast::getVolumeArea(boundaryVolume);			
			//_app._volume->emplaceChannel(std::move(boundaryVolume));
			//_app._volume->emplaceChannel(std::move(areas));r

			
			auto res = getVolumeAreaMesh(boundaryVolume);
			
			if (res.Nverts > 0) {
				_app._volumeMC = std::move(VertexBuffer<VertexData>(res.vbo, res.Nverts));
			}
			else {
				_app._volumeMC = std::move(VertexBuffer<VertexData>());
			}

		}

		auto rad = fast::getReactiveAreaDensityTensor<double>(ccl);

		for (auto i = 0; i < 6; i++) {
			std::cout << "RAD (dir: " << i << "): " << rad[i] << std::endl;
		}

	}
	ImGui::SameLine();
	ImGui::Checkbox("Gen. Mesh", &enableRADMesh);

	static bool saveMeshCCL = false;
	ImGui::Checkbox("Save Mesh CCL", &saveMeshCCL);
	static bool saveMeshCCLDir = false;
	ImGui::Checkbox("CCL Dir", &saveMeshCCLDir);

	if (ImGui::Button("Save .bin")) {
		int channelID = _app._options["Render"].get<int>("channel");
		auto it = std::next(_app._volumes.begin(), channelID);
		auto & renderChannel = *(it->second);
		fast::saveVolumeBinary("volume.bin", renderChannel);
	}

	if (ImGui::Button("Save mesh") /*&& _app._volumeMC.size() > 0*/) {
		auto ccl = fast::getVolumeCCL(*_app._volumes[CHANNEL_MASK], 255);


		//auto vols = generateSeparatedCCLVolumes(ccl);				
		

		std::map<std::string, std::unique_ptr<fast::Volume>> vols;

		vols["volume"] = std::make_unique<fast::Volume>(_app._volumes[CHANNEL_MASK]->clone());		
		if (saveMeshCCL) {
			vols["cclAll"] = std::make_unique<fast::Volume>(fast::generateBoundaryConnectedVolume(ccl, DIR_NONE, false));
			vols["cclSub"] = std::make_unique<fast::Volume>(fast::generateBoundaryConnectedVolume(ccl, DIR_NONE, true));			
		}

		if (saveMeshCCLDir) {			

			vols["cclAll"] = std::make_unique<fast::Volume>(fast::generateBoundaryConnectedVolume(ccl, DIR_NONE, false));
			vols["cclSub"] = std::make_unique<fast::Volume>(fast::generateBoundaryConnectedVolume(ccl, DIR_NONE, true));
			auto & vall = *vols["cclSub"];
			for (auto k = 0; k < 3; k++) {
				
				Dir dpos = getDir(k, +1);
				Dir dneg = getOppositeDir(dpos);

				auto vpos = fast::generateBoundaryConnectedVolume(ccl, dpos, true);
				auto vneg = fast::generateBoundaryConnectedVolume(ccl, dneg, true);
				vpos.getPtr().retrieve();
				vneg.getPtr().retrieve();

				auto subpos = vall.op(vneg, fast::Volume::VO_SUB);
				auto subneg = vall.op(vpos, fast::Volume::VO_SUB);	
				auto submid = vneg.op(vpos, fast::Volume::VO_MIN);
				//vall.op(vpos, fast::Volume::VO_SUB).op(vneg, fast::Volume::VO_SUB);
				
				vols[toString("cclSubDir%Mid", k)] = std::make_unique<fast::Volume>(std::move(submid));
				vols[toString("cclSubDir%Pos", k)] = std::make_unique<fast::Volume>(std::move(subpos));
				vols[toString("cclSubDir%Neg", k)] = std::make_unique<fast::Volume>(std::move(subneg));				
			}
		}


		for (auto & it : vols) {			
			it.second = std::make_unique<fast::Volume>(it.second->withZeroPadding(ivec3(1), ivec3(1)));
			auto & vol = *it.second;

			vol.getPtr().createTexture();
			auto res = fast::getVolumeAreaMesh(vol);
			

			if (res.Nverts > 0) {
				fast::CUDA_VBO _tmp(res.vbo);
				_tmp.saveObj((it.first + ".obj").c_str());
				std::cout << "saved " << it.first << " vbo " << res.vbo << " N " << res.Nverts << std::endl;
			}
			else {
				std::cout << it.first << " has no triangles" << std::endl;
			}

			_app._volumes[it.first] = std::move(it.second);
		}

		/*{
			auto vol = fast::generateBoundaryConnectedVolume(ccl, DIR_NONE);
			
			vol.getPtr().createTexture();						
			fast::getVolumeAreaMesh(vol, &vboIndex, &Nverts);
			fast::CUDA_VBO _tmp(vboIndex);
			_tmp.saveObj("cclAll.obj");
			std::cout << "saved cclAll vbo " << vboIndex << " N " << Nverts << std::endl;
			_app._volumes["cclAll"] = std::make_unique<fast::Volume>(std::move(vol));
			
		}
		{
			auto vol = fast::generateBoundaryConnectedVolume(ccl, DIR_NONE, true);
			
			vol.getPtr().createTexture();
			fast::getVolumeAreaMesh(vol, &vboIndex, &Nverts);
			
			if (Nverts > 0) {
				fast::CUDA_VBO _tmp(vboIndex);
				_tmp.saveObj("cclSub.obj");
				std::cout << "saved cclSub vbo " << vboIndex << " N " << Nverts << std::endl;
			}
			_app._volumes["cclSub"] = std::make_unique<fast::Volume>(std::move(vol));
		}
*/

		/*for(auto i=0; i < vols.size(); i++){
			std::string name = (i == 0) ? "background" : toString("ccl%", i);
			_app._volumes[name] = std::move(vols[i]);
		}*/


		/*fast::CUDA_VBO _tmp(_app._volumeMC.getVBOIndex());
		_tmp.saveObj("mesh.obj");				*/
	}
	



	static bool concGraph = false;
	ImGui::Checkbox("Concetration Graph", &concGraph);
	
	if(concGraph){

		ImGui::SameLine();

		
		const Dir dir = Dir(_app._options["Diffusion"].get<int>("direction"));		
		
		static std::vector<double> vals;	


		auto & c = *_app._volumes[CHANNEL_CONCETRATION];
		assert(c.type() == TYPE_DOUBLE);
		
		if (ImGui::Button("Refresh")) {
			vals.resize(c.dimInDirection(dir), 0.0f);
			c.sumInDir(dir, vals.data());

			auto sliceElemCount = float(c.sliceElemCount(dir));
			for (auto & v : vals)
				v /= sliceElemCount;

		}

		{
			std::vector<float> tmp;
			for (auto f : vals) tmp.push_back(float(f));

			ImGui::PlotLines("C", tmp.data(), int(tmp.size()), 0, nullptr, 0.0f, 1.0f, ImVec2(200, 300));
		}	
	}

	ImGui::Separator();

	{
		if (ImGui::Button("Load Default")) {
			_app.loadFromFile(_app._options["Input"].get<std::string>("DefaultPath"));
		}

		ImGui::SameLine();

		if (ImGui::Button("Reset")) {
			_app.reset();
		}

	}

	//static bool packingOpen = true;
	static int packN = 1;
	static int rasterResViz = 32;
	static int rasterResOnce = 128;
	static vec3 packSize = vec3(0.5f);
	static float targetPorosity = 0.6f;
	static vec3 domainMin = vec3(-1.0f);
	static vec3 domainMax = vec3(1.0f);
	static bool packShowBest = false;
		

	auto checkVolumeSize = [&](bool once) {
		int res = (once) ? rasterResOnce : rasterResViz;

		vec3 range = domainMax - domainMin;
		float maxSide = glm::max(range.x, glm::max(range.y, range.z));

		ivec3 res3 = {
			int(res * (range.x / maxSide)),
			int(res * (range.y / maxSide)),
			int(res * (range.z / maxSide))
		};

		if (res != _app._volumes[CHANNEL_MASK]->dim().x) {
			_app._volumes[CHANNEL_MASK] = std::make_unique<fast::Volume>(res3, TYPE_UCHAR);
			_app._volumes[CHANNEL_MASK]->getPtr().createTexture();
			_app._sdfpacking->setRasterVolume(_app._volumes[CHANNEL_MASK].get());

		}
	};


	if (ImGui::CollapsingHeader("Packing", ImGuiTreeNodeFlags_DefaultOpen)) {


		if (ImGui::Button("Pack")){			
			_app._sdfpacking = std::make_unique<fast::SDFPacking>(fast::AABB(domainMin,domainMax));
			checkVolumeSize(false);
			_app._options["Render"].get<int>("channel") = 0;
			_app._sdfpacking->setRasterVolume(_app._volumes[CHANNEL_MASK].get());			
			_app._sdfpacking->addEllipse(packN, packSize);
			_app._sdfpacking->init();
			
		} ImGui::SameLine();


		

		if (ImGui::Button("eps-calc")) {
			float v0 = packSize.x * packSize.y * packSize.z * (4.0f / 3.0f) * glm::pi<float>();
			auto domain= fast::AABB(domainMin, domainMax);

			float v0rel = v0 / domain.volume();
			float targetVolume = (1.0f - targetPorosity) * domain.volume();
			int N0 = targetVolume / v0;

			int N = (1.0f - targetPorosity) / v0rel;
			targetPorosity = (1.0f - N * v0rel);
			packN = N;						

		}
		
		ImGui::Checkbox("Auto step", &_app._sdfpackingAutoUpdate);
		ImGui::Checkbox("Show Best", &packShowBest);
		ImGui::InputInt("PackN", &packN);		
		ImGui::InputFloat3("PackSize", ((float*)&packSize));
		ImGui::InputFloat3("DomainMin", ((float*)&domainMin));
		ImGui::InputFloat3("DomainMax", ((float*)&domainMax));
		ImGui::InputInt("RasterRes", &rasterResViz);
		ImGui::InputInt("RasterResOnce", &rasterResOnce);
		ImGui::DragFloat("TargetPorosity", &targetPorosity, 0.01f, 0.0f, 1.0f);		



		if (ImGui::Button("Step SA")) {
			checkVolumeSize(false);
			if (_app._sdfpacking && !_app._sdfpacking->step()) {
			}
		}

		ImGui::SameLine();
		if (ImGui::Button("Raster Once")) {
			checkVolumeSize(true);
			_app._sdfpacking->rasterize();
		}


		if (_app._sdfpacking) {
			auto & vec = _app._sdfpacking->getSA().scoreHistory;
			float minVal = FLT_MAX;
			float maxVal = -FLT_MAX;
			for (auto f : vec) {
				minVal = glm::min(minVal, f);
				maxVal = glm::max(maxVal, f);
			}
			ImGui::PlotLines("C", vec.data(), int(vec.size()), 0, nullptr, minVal, maxVal, ImVec2(200, 300));
		}	
	}

	if (_app._sdfpacking != nullptr && _app._sdfpackingAutoUpdate) {

		
		_app._sdfpacking->setShowBest(packShowBest);

		checkVolumeSize(false);		

		if (_app._sdfpacking->getSA().rejections > 100) {
			std::cout << "FINISHED" << std::endl;
			_app._sdfpackingAutoUpdate = false;
		}

		static int _cnt = 0;
		
		if(_cnt % 15 == 0)
			std::cout << "Penetration: " << _app._sdfpacking->getMaxPenetrationFraction(3) << std::endl;
		_cnt++;

		if (!_app._sdfpacking->step()) {			
		}
	}


	if (ImGui::CollapsingHeader("Load .TIFF")) {		

		//static std::string curDir = "../../data";
		static std::string curDir = "D:/!battery/datasetAll";

		//static std::string curDir = "D:/!battery";
		std::string filename;
		std::tie(curDir, filename) = imguiFileExplorer(curDir, "tiff", true);
		if (filename != "") {
			std::cout << "Loading " << filename << std::endl;
			_app.loadFromFile(curDir);
		}		
	}

	if (ImGui::CollapsingHeader("Load .sph")) {
		static ivec3 resolution = { 128,128,128 };
		static std::string curDir = "../../data";
		ImGui::InputInt3("Sphere Resolution", (int*)&resolution);
		//static std::string curDir = "D:/!battery";
		std::string filename;
		std::tie(curDir, filename) = imguiFileExplorer(curDir, "sph", true);
		if (filename != "") {
			std::cout << "Loading " << filename << std::endl;

			std::ifstream f(filename);
			auto spheres = fast::loadSpheres(f);
			_app.loadFromMask(fast::rasterizeSpheres(resolution, spheres));			
		}
	}

	if (ImGui::CollapsingHeader("Load .bin")) {
		//static ivec3 resolution = { 128,128,128 };
		static std::string curDir = "D:/!battery/";//"../";
		//ImGui::InputInt3("Sphere Resolution", (int*)&resolution);
		//static std::string curDir = "D:/!battery";
		std::string filename;
		std::tie(curDir, filename) = imguiFileExplorer(curDir, "bin", true);
		if (filename != "") {
			std::cout << "Loading " << filename << std::endl;
			std::ifstream f(filename);
			auto vol = fast::loadVolumeBinary(filename.c_str());
			_app.loadFromMask(std::move(vol));
		}
	}

	if (ImGui::CollapsingHeader("Load .pos")) {
		static std::string curDir = "../../data/shapes/";
		static int currentIndex = 0;
		const float scale = 1.0f / glm::pow(3.0f, 1.0f / 3.0f);
		static fast::AABB bb = { vec3(0), vec3(scale)};	
			

		std::string filename;
		std::tie(curDir, filename) = imguiFileExplorer(curDir, "pos", true);
		if (filename != "") {			

			std::cout << "Loading " << filename << std::endl;
			_app.loadFromPosFile(filename, ivec3(_app._options["Generator"].get<int>("Resolution")), currentIndex, bb);
		}
		ImGui::InputInt("Index in .pos", &currentIndex);
		ImGui::InputFloat3("Min BB", reinterpret_cast<float*>(&bb.min));		
		ImGui::InputFloat3("Max BB", reinterpret_cast<float*>(&bb.max));
	}

	ImGui::Separator();
	ImGui::Text("Generate volume");
	ImGui::Separator();

	


	renderOptionSet("Generator Options", _app._options["Generator"], 0);

	int genResX = _app._options["Generator"].get<int>("Resolution");
	ivec3 genRes = ivec3(genResX);
	//ivec3 genRes = ivec3(genResX, genResX/2, genResX/4);

	if (ImGui::Button("Spheres")) {
		auto & opt = _app._options["Generator"]["Spheres"];

		fast::GeneratorSphereParams p;
		p.N = opt.get<int>("N");
		p.rmin = opt.get<float>("RadiusMin");
		p.rmax = opt.get<float>("RadiusMax");
		p.maxTries = opt.get<int>("MaxTries");
		p.overlapping = opt.get<bool>("Overlapping");
		p.withinBounds = opt.get<bool>("WithinBounds");


		
		auto spheres = fast::generateSpheres(p);

		std::cout << "Analytic tau: " << fast::spheresAnalyticTortuosity(p, spheres);
		
		_app.loadFromMask(
			fast::rasterizeSpheres(genRes, spheres)
		);

		std::cout << fast::getPorosity<double>(*_app._volumes[CHANNEL_MASK]) << std::endl;

		
	}

	ImGui::SameLine();
	if (ImGui::Button("Maximize N Spherers")) {
		auto & opt = _app._options["Generator"]["Spheres"];

		fast::GeneratorSphereParams p;
		p.N = opt.get<int>("N");
		p.maxTries = opt.get<int>("MaxTries");
		p.rmin = opt.get<float>("RadiusMin");
		p.rmax = opt.get<float>("RadiusMax");		
		p.withinBounds = opt.get<bool>("WithinBounds");
		p.overlapping = false;

		
		while (true) {

			bool result = true;
			result = _app.loadFromMask(
				fast::rasterizeSpheres(genRes, fast::generateSpheres(p))
			);

			if (!result) break;
			p.N += 1;
		}

		
		std::cerr << "Max N " << p.N << std::endl;		
	}

	if (ImGui::Button("Maximize R Spherers")) {
		fast::GeneratorSphereParams p;
		auto & opt = _app._options["Generator"]["Spheres"];
		p.N = opt.get<int>("N");
		p.maxTries = opt.get<int>("MaxTries");		
		p.withinBounds = opt.get<bool>("WithinBounds");
		p.overlapping = false;

		float r = fast::findMaxRandomSpheresRadius(p,0.0001f);
		std::cout << "max r: " << r << std::endl;
		opt.set<float>("RadiusMin", r);
		opt.set<float>("RadiusMax", r);

		

		p.rmax = r;
		p.rmin = r;		
		p.maxTries = -1;

		_app.loadFromMask(
			fast::rasterizeSpheres(genRes, fast::generateSpheres(p))
		);
	}

	if (ImGui::Button("Generate max R's")) {


		fast::GeneratorSphereParams p;
		auto & opt = _app._options["Generator"]["Spheres"];
		
		p.maxTries = opt.get<int>("MaxTries");
		p.withinBounds = opt.get<bool>("WithinBounds");
		p.overlapping = false;

		std::ofstream f("maxRspheres.txt");

		/*std::vector<ivec3> resolutions;

		for (int i = 32; i <= 256; i+=32) {
			resolutions.push_back(ivec3(i));
		}*/

		for (p.N = 100; p.N <= 2000; p.N+=10) {

			p.maxTries = opt.get<int>("MaxTries");
			float r = fast::findMaxRandomSpheresRadius(p, 0.0001f);
			std::cout << "N: "<< p.N << ",  max r: " << r << std::endl;
					
			f << p.N << "\t" << r << "\n";
			f.flush();

			/*p.rmax = r;
			p.rmin = r;
			p.maxTries = -1;*/

			//char buf[512];
			
			/*auto spheres = fast::generateSpheres(p);

			for (auto res : resolutions) {
				sprintf(buf, "spheres_%d_%f_%d.vol", p.N, r, res.x);
				
				auto vol = fast::rasterizeSpheres(res, spheres);
				std::cout << "Saving to " << buf << std::endl;
				fast::saveVolumeBinary(buf, vol);
			}*/
			
			
			
		}

	
	}

	if (ImGui::Button("Filled")) {
		auto & opt = _app._options["Generator"]["Filled"];		
		_app.loadFromMask(
			fast::generateFilledVolume(genRes, uchar(opt.get<int>("value")))
		);
	}


	ImGui::Separator();

	static int backgroundValue = 0;
	ImGui::InputInt("Background CCL", &backgroundValue, 8, 255);


	if (ImGui::Button("CCL")) {

		
		auto ccl = getVolumeCCL(*_app._volumes[CHANNEL_MASK], backgroundValue);
		
		for (auto i = 0; i < 6; i++) {
			auto vol = generateBoundaryConnectedVolume(ccl, Dir(i));			
			
			_app._volumes[toString("ccl%",i)] = std::make_unique<fast::Volume>(
				std::move(generateCCLVisualization(ccl, &vol))
				);
			_app._volumes[toString("cclvol%", i)] = std::make_unique<fast::Volume>(std::move(vol));
			//_app._volume->emplaceChannel(std::move(vol));
			//_app._volume->emplaceChannel(generateBoundaryConnectedVolume(ccl, Dir(i)));
		}

		//_app._volume->emplaceChannel(generateCCLVisualization(ccl));
		

	}


	ImGui::Separator();

	if (ImGui::Button("Screenshot")) {
		auto cam = _app._camera;
		cam.setWindowDimensions(1920, 1080);
		_app._volumeRaycaster->saveScreenshot(
			cam,
			ivec4(0, 0, cam.getWindowWidth(), cam.getWindowHeight()),
			("screenshot_" + timestampString() + ".png").c_str()
		);
	}

	static int videoStep = 5;

	if (ImGui::Button("ScreenshotVideo")) {
		auto cam = _app._camera;
		cam.setWindowDimensions(1920, 1080);


		
		fast::TortuosityParams tp;
		tp.coeffs = {
			_app._options["Diffusion"].get<float>("D_zero"),
			_app._options["Diffusion"].get<float>("D_one")
		};
		tp.dir = Dir(_app._options["Diffusion"].get<int>("direction"));
		tp.tolerance = powf(10.0, -_app._options["Diffusion"].get<int>("Tolerance"));
		//tp.maxIter = size_t(_app._options["Diffusion"].get<int>("maxIter"));		
		tp.coeffs[1] = powf(10.0, -_app._options["Diffusion"].get<int>("expD_one"));
		tp.verbose = tauVerbose;		
		tp.useNonUniform = true;

		auto & mask = *_app._volumes[CHANNEL_MASK];
		fast::Volume * outputChannel = nullptr;
		if (_app._volumes.find(CHANNEL_CONCETRATION) != _app._volumes.end()) {
			outputChannel = _app._volumes[CHANNEL_CONCETRATION].get();
		}

		tp.maxIter = 1;
		int step = videoStep;

		float tau = 0.0f;
		int iter = 0;
		while (tau == 0.0f) {
			tau = fast::getTortuosity<float>(mask, tp, fast::DiffusionSolverType::DSOLVER_BICGSTAB, outputChannel);

			char buf[64];
			sprintf(buf, "%03d", int(iter));
			
			_app._volumeRaycaster->saveScreenshot(
				cam,
				ivec4(0, 0, cam.getWindowWidth(), cam.getWindowHeight()),
				("video/screenshot_" + std::string(buf) + ".png").c_str()
			);
			iter++;
			tp.maxIter += step;
		}

	}

	ImGui::SameLine();
	ImGui::InputInt("Step", &videoStep);





	ImGui::End();


	ImGui::Render();
}

void Ui::callbackMouseButton(GLFWwindow * w, int button, int action, int mods)
{
	ImGui_ImplGlfwGL3_MouseButtonCallback(w, button, action, mods);
}

void Ui::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{
	
	ImGui_ImplGlfwGL3_KeyCallback(w, key, scancode, action, mods);
}

void Ui::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{
	ImGui_ImplGlfwGL3_ScrollCallback(w, xoffset, yoffset);
}

void Ui::callbackChar(GLFWwindow * w, unsigned int code)
{
	
	ImGui_ImplGlfwGL3_CharCallback(w, code);
}

bool Ui::isFocused() const
{
	return ImGui::IsAnyItemActive();
}

