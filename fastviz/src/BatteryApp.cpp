#include "BatteryApp.h"
#include "GLFW/glfw3.h"

#include "utility/IOUtility.h"



#include "render/MeshObject.h"

#include "render/VolumeRaycaster.h"
#include "render/Shader.h"

#include <fastlib/volume/VolumeIO.h>
#include <fastlib/utility/RandomGenerator.h>
#include <fastlib/geometry/GeometryIO.h>
#include <fastlib/volume/VolumeGenerator.h>
#include <fastlib/optimization/SDFPacking.h>


#include <glm/gtc/matrix_transform.hpp>




#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <array>
#include <numeric>






using namespace std;
using namespace fast;

RNGNormal normalDist(0, 1);
RNGUniformFloat uniformDist(0, 1);
RNGUniformInt uniformDistInt(0, INT_MAX);



#include "render/PrimitivesVBO.h"
#include "render/Shaders.h"


#include <fastlib/utility/Timer.h>

#include <fastlib/volume/VolumeMeasures.h>
#include <fastlib/volume/VolumeRasterization.h>
#include <fastlib/geometry/GeometryIO.h>
#include <fastlib/geometry/GeometryObject.h>
#include <fastlib/geometry/SAP.h>
#include <fastlib/geometry/Intersection.h>




BatteryApp::BatteryApp()
	: App("BatteryViz"),
	_camera(Camera::defaultCamera(_window.width, _window.height)),	
	_ui(*this),
	_currentRenderChannel(0),
	_sdfpackingAutoUpdate(true)
{	


	
	fast::Volume::enableOpenGLInterop = true;

	{
		std::ifstream optFile(OPTIONS_FILENAME);
		if (optFile.good())
			optFile >> _options;
		else
			throw string("Options file not found");
	}

	resetGL();
	
	
	{
		auto errMsg = loadShaders(_shaders);
		if (errMsg != "")
			throw errMsg;
	}

	_volumeRaycaster = make_unique<VolumeRaycaster>(
		_shaders[SHADER_POSITION],
		_shaders[SHADER_VOLUME_RAYCASTER],
		_shaders[SHADER_VOLUME_SLICE]
	);
	
		
	
	_autoUpdate = false;

	_aabbVBO = getCubeVBO();
	
	

	bool res = loadFromFile(_options["Input"].get<std::string>("DefaultPath"));
	if (!res){
		std::cerr << "Failed to load default path" << std::endl;
		reset();
	}
	

	

	//INIT
	//reset();

	/*
		Scene init
	*/

/*
	auto sphereObj = make_shared<MeshObject>(fast::generateSphere());
	_scene.addObject("sphere", sphereObj);
*/

	


	
	

}





void BatteryApp::update(double dt)
{
	
	{
		static float iso = _options["Render"].get<float>("MarchingCubesIso");
		static int res = _options["Render"].get<int>("MarchingCubesRes");		
		static float smooth = _options["Render"].get<float>("MarchingCubesSmooth");

		float newIso = _options["Render"].get<float>("MarchingCubesIso");
		float newSmooth = _options["Render"].get<float>("MarchingCubesSmooth");
		int newRes = _options["Render"].get<int>("MarchingCubesRes");
		if (newRes < 8) newRes = 8;

		if (newIso != iso || newRes != res || newSmooth != smooth) {
			res = newRes;
			iso = newIso;
			smooth = newSmooth;
			runAreaDensity();
		}		


	}

	

	

	
	
	
	if (!_autoUpdate) return;





	return;
}

void BatteryApp::render(double dt)
{
	

	if (_window.width == 0 || _window.height == 0) return;

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	
	
	float sliceHeight = 0;
	if (_options["Render"].get<bool>("slices")) {
		sliceHeight = 1.0f / 3.0f;
	}
	

	if (_options["Render"]["Packing"].get<bool>("BB") && _sdfpacking){
		RenderList rl;

		Transform volumeTransform;
		volumeTransform.scale = vec3(2);
		volumeTransform.translation = vec3(-1);


		auto aabbs = _sdfpacking->getParticleBounds();

		for (auto bb : aabbs){

			Transform tbb;
			tbb.scale = vec3(0.5f) * bb.range();
			tbb.translation = bb.centroid();
			

			{
				mat4 M = /*volumeTransform.getAffine() **/ tbb.getAffine();
				mat4 NM = mat4(glm::transpose(glm::inverse(mat3(M))));

				ShaderOptions so = {
					{ "M", M },
					{ "NM", NM },
					{ "PV", _camera.getPV() },
					{ "useUniformColor", true },
					{ "uniformColor", vec4(1.0f,0,0,1.0f) }
				};
				RenderList::RenderItem item = { _aabbVBO, so, GL_LINE };
				rl.add(_shaders[SHADER_FLAT], item);
			}
		}

		rl.render();
	}
	

	
	if (_options["Render"]["SceneGeometry"].get<bool>("Enabled")) {

		RenderList rl;

		bool bboxes = _options["Render"]["SceneGeometry"].get<bool>("BoundingBoxes");

		

		
		for (auto & p : _sceneGeometry) {
			
			Transform volumeTransform;
			volumeTransform.scale = vec3(2);
			volumeTransform.translation = vec3(-1);

			auto templateGeometry = p->getTemplateGeometry();
			auto vboIt = _geometryVBOs.find(templateGeometry);
			if (vboIt != _geometryVBOs.end()) {

				auto & vbo = vboIt->second;

				auto t = p->getTransform();
				{
					mat4 M = volumeTransform.getAffine() * t.getAffine();
					mat4 NM = mat4(glm::transpose(glm::inverse(mat3(M))));

					ShaderOptions so = { { "M", M },{ "NM", NM },{ "PV", _camera.getPV() },{ "viewPos", _camera.getPosition() } };
					RenderList::RenderItem item = { vbo, so, GL_FILL };
					rl.add(_shaders[SHADER_PHONG], item);
				}
			}

			if (bboxes) {
				//auto t = p->getTransform();
				auto bb = p->bounds();

				Transform tbb;
				tbb.scale = vec3(0.5f) * bb.range();
				tbb.translation = bb.centroid();
				
				{
					mat4 M = volumeTransform.getAffine() * tbb.getAffine();
					mat4 NM = mat4(glm::transpose(glm::inverse(mat3(M))));

					ShaderOptions so = { 
						{ "M", M },
						{ "NM", NM },
						{ "PV", _camera.getPV() },
						{ "useUniformColor", true},
						{ "uniformColor", vec4(1.0f,0,0,1.0f) }
					};
					RenderList::RenderItem item = { _aabbVBO, so, GL_LINE };
					rl.add(_shaders[SHADER_FLAT], item);
				}
			
			}

		}

		

		rl.render();

	}




	/*
		Gl renderer
	*/
	if (_options["Render"].get<bool>("scene")) {
		auto renderList = getSceneRenderList(_scene, _shaders, _camera);
		renderList.render();	
	}

	/*if (_options["Render"].get<bool>("ellipsoids")) {

		RenderList rl;

		static auto vboSphere = getSphereVBO();	

		for (auto & e : _saEllipsoid.state) {
			auto T = e.transform.getAffine();
			{
				mat4 M = *reinterpret_cast<const mat4*>(T.data());
				mat4 NM = mat4(glm::transpose(glm::inverse(mat3(M))));

				ShaderOptions so = { { "M", M },{ "NM", NM },{ "PV", _camera.getPV() },{ "viewPos", _camera.getPosition() } };
				RenderList::RenderItem item = { vboSphere, so };
				rl.add(_shaders[SHADER_PHONG], item);
			}
		}

		rl.render();		
	}

	if (_options["Render"].get<bool>("ellipsoidsBounds")) {
		RenderList rl;

		static auto vboCube = getCubeVBO();

		for (auto & e : _saEllipsoid.state) {
			auto bounds = e.aabb();
			
			EigenTransform boxT;
			//Cube VBO is from -1 to 1
			boxT.scale = (bounds.max - bounds.min);
			boxT.translation = (bounds.max + bounds.min) * 0.5f;


			mat4 M = *reinterpret_cast<const mat4*>(boxT.getAffine().data()) *
				glm::scale(mat4(1.0f), vec3(0.5f));
			ShaderOptions so = { { "M", M },{ "NM", M },{ "PV", _camera.getPV() },
				{ "useUniformColor", true },{ "uniformColor", vec4(1,0,0,1) }
			};
			RenderList::RenderItem item = { vboCube, so, GL_LINE };
			rl.add(_shaders[SHADER_FLAT], item);
			
		}

		rl.render();
	
	}*/


	

	/*
	Volume raycaster
	*/
	ivec4 viewport = {
		0, _window.height * sliceHeight, _window.width, _window.height - _window.height * sliceHeight
	};

	if (_options["Render"].get<bool>("volume") && _volumes.size() > 0) {
		
		//update current channel to render
		_currentRenderChannel = _options["Render"].get<int>("channel");
		
		_currentRenderChannel = std::min(std::max(_currentRenderChannel, uint(0)), uint(_volumes.size() - 1));
		_options["Render"].set<int>("channel", _currentRenderChannel);
		_volumeRaycaster->setVolume(*std::next(_volumes.begin(), _currentRenderChannel)->second);
		

		if (_options["Render"].get<bool>("transferDefault")) {
			if (std::next(_volumes.begin(), _currentRenderChannel)->second->type() == TYPE_UCHAR)
				_volumeRaycaster->setTransferGray();
			else
				_volumeRaycaster->setTransferJet();
		}
		else {
			int t = _options["Render"].get<int>("transferFunc");
			if(t == 0)
				_volumeRaycaster->setTransferGray();
			else
				_volumeRaycaster->setTransferJet();
		}

		_volumeRaycaster->enableFiltering(
			_options["Render"].get<bool>("volumeFiltering")
		);

		
		
			

		_camera.setWindowDimensions(_window.width, _window.height - static_cast<int>(_window.height * sliceHeight));	

		_volumeRaycaster->render(_camera, viewport);	

	}

	if (_options["Render"].get<bool>("volumeGrid")) {
		_volumeRaycaster->renderGrid(_camera, viewport, *_shaders[SHADER_FLAT],
			_options["Render"].get<float>("volumeGridOpacity")

		);
	}

	//Render marching cubes volume
	{
		RenderList rl;

		mat4 M = glm::scale(mat4(1.0f), vec3(2.0f));
		ShaderOptions so = {
			{ "M", M },
			{ "NM", M },
			{ "PV", _camera.getPV() }			
		};


		const GLenum fill = _options["Render"].get<bool>("MarchingCubesWire") ? GL_LINE : GL_FILL;

		RenderList::RenderItem item = { _volumeMC, so, fill, GL_BACK };
		rl.add(_shaders[SHADER_PHONG], item);
		rl.render();
	}

	
	/*
	Volume slices
	*/	
	if (_options["Render"].get<bool>("slices")) {
		mat3 R[3] = {
			mat3(),
			mat3(glm::rotate(mat4(1.0f), glm::radians(90.0f), vec3(0, 0, 0))),
			mat3(glm::rotate(mat4(1.0f), glm::radians(90.0f), vec3(1, 0, 0)))
		};

		int yoffset = 0;
		int ysize = static_cast<int>(_window.height  * sliceHeight);

		_volumeRaycaster->renderSlice(0, ivec2(_window.width / 3 * 0, yoffset), ivec2(_window.width / 3, ysize));
		_volumeRaycaster->renderSlice(1, ivec2(_window.width / 3 * 1, yoffset), ivec2(_window.width / 3, ysize));
		_volumeRaycaster->renderSlice(2, ivec2(_window.width / 3 * 2, yoffset), ivec2(_window.width / 3, ysize));
	}


	/*
		UI render and update
	*/
	_ui.update(dt);
}



void BatteryApp::runAreaDensity()
{
	/*uint vboIndex;
	size_t Nverts = 0;

	float iso = _options["Render"].get<float>("MarchingCubesIso");
	int res = _options["Render"].get<int>("MarchingCubesRes");
	auto & mask = _volume->getChannel(CHANNEL_MASK);
	//float smooth = float(res) / mask.dim().x;
	
	//ivec3 mcres = mask.dim();
	ivec3 mcres = ivec3(res);

	double a = fast::getReactiveAreaDensity<double>(mask, mcres, iso, 1.0f, &vboIndex, &Nverts);

	int nparticles = _options["Generator"]["Spheres"].get<int>("N");
	double porosity = fast::getPorosity<double>(_volume->getChannel(CHANNEL_MASK));
	double volume = 1.0 - porosity;
	std::cout << "Porosity : " << porosity << ", Particle volume: " << volume << std::endl;
	double sf = fast::getShapeFactor(a / nparticles, volume / nparticles);

	size_t N = mask.dim().x * mask.dim().y  *mask.dim().z;

	std::cout << "Reactive Area Density: " << a << ", Shape Factor: " << sf << " per " << nparticles << " particles, normalized a: " << a / N << "\n";	
	
	if (Nverts > 0) {
		_volumeMC = std::move(VertexBuffer<VertexData>(vboIndex, Nverts));
	}
	else {
		_volumeMC = std::move(VertexBuffer<VertexData>());
	}*/
}




void BatteryApp::callbackMousePos(GLFWwindow * w, double x, double y)
{
	App::callbackMousePos(w, x, y);

	if (_input.mouseButtonPressed[GLFW_MOUSE_BUTTON_2]) {

		auto & cam = _camera;

		glm::vec2 angle = (_input.mousePos - _input.mouseButtonPressPos[GLFW_MOUSE_BUTTON_2]) / 360.0f;
		std::swap(angle.x, angle.y);
		angle.y = -angle.y;

		auto sideAxis = glm::cross(cam.getUp(), cam.getDirection());

		glm::vec4 cpos = glm::vec4(cam.getPosition() - cam.getLookat(), 1.0f);
		glm::mat4 R1 = glm::rotate<float>(glm::mat4(1.0f), angle.y, cam.getUp());
		glm::mat4 R0 = glm::rotate<float>(glm::mat4(1.0f), angle.x, sideAxis);
		cpos = R1 * R0  * cpos;;
		cpos += glm::vec4(cam.getLookat(), 0.0f);
		
		cam.setPosition(glm::vec3(cpos.x, cpos.y, cpos.z));

		_input.mouseButtonPressPos[GLFW_MOUSE_BUTTON_2] = _input.mousePos;	
	}

}

void BatteryApp::callbackMouseButton(GLFWwindow * w, int button, int action, int mods)
{
	
	App::callbackMouseButton(w, button, action, mods);
	_ui.callbackMouseButton(w, button, action, mods);
		
}

void BatteryApp::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{

	if (_ui.isFocused()) {
		_ui.callbackKey(w, key, scancode, action, mods);
		return;
	}


	App::callbackKey(w, key, scancode, action, mods);
		

	if (action == GLFW_RELEASE || action == GLFW_REPEAT) {
		
		if (key == GLFW_KEY_R) {
			std::cout << "Reloading shaders ...";
			std::cerr << loadShaders(_shaders) << std::endl;			
			std::cout << "Done." << std::endl;
		}

		if (key == GLFW_KEY_SPACE)
			_autoUpdate = !_autoUpdate;

		if (key == GLFW_KEY_Q)
			reset();


		if (key == GLFW_KEY_1) {
			_options["Render"].get<int>("channel") = 0;
		}

		if (key == GLFW_KEY_2 && _volumes.size() > 1) {
			_options["Render"].get<int>("channel") = 1;
		}
		if (key == GLFW_KEY_3 && _volumes.size() > 2) {
			_options["Render"].get<int>("channel") = 2;
		}
		if (key == GLFW_KEY_4 && _volumes.size() > 3) {
			_options["Render"].get<int>("channel") = 3;
		}

		if (key == GLFW_KEY_RIGHT) {
			_options["Render"].get<int>("channel")++;
			
			if (_options["Render"].get<int>("channel") >= int(_volumes.size()))
				_options["Render"].get<int>("channel") = 0;

		}

		if (key == GLFW_KEY_LEFT) {
			_options["Render"].get<int>("channel")--;
			if (_options["Render"].get<int>("channel") < 0)
				_options["Render"].get<int>("channel") = int(_volumes.size()) - 1;
		}

	}

	
}

void BatteryApp::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{
	App::callbackScroll(w, xoffset, yoffset);
	
	auto & cam = _camera;
	auto delta = static_cast<float>(0.1 * yoffset);
	cam.setPosition(((1.0f - 1.0f*delta) * (cam.getPosition() - cam.getLookat())) + cam.getLookat());
	
	_ui.callbackScroll(w, xoffset, yoffset);
}

void BatteryApp::callbackChar(GLFWwindow * w, unsigned int code)
{
	App::callbackChar(w, code);
	_ui.callbackChar(w, code);
}

bool BatteryApp::loadFromFile(const std::string & folder)
{

	//ivec3 userSize = ivec3(_options["Input"].get<int>("GenResolution"));
	ivec3 userSize = ivec3(_options["Input"].get<ivec3>("LoadResolution"));
	ivec3 origin = ivec3(_options["Input"].get<ivec3>("LoadOrigin"));
	if (userSize.x == 0) {
		userSize = ivec3(128);
	}

	try {

		if(checkExtension(folder,"sph")){
			std::ifstream f(folder);
			auto spheres = fast::loadSpheres(f);
			Volume c = fast::rasterizeSpheres(userSize, spheres);
				
			
			return loadFromMask(std::move(c));	

		}
		else{

			Volume c = loadTiffFolder(folder.c_str());

			
			ivec3 size = glm::min(c.dim(), userSize);
			c.resize(ivec3(origin), ivec3(size));

			return loadFromMask(std::move(c));	
		}	
	}
	catch (const char * msg) {
		std::cerr << msg << std::endl;
		return false;
	}

	return true;
}


bool BatteryApp::loadFromPosFile(const std::string & path, ivec3 resolution, size_t index, const fast::AABB & trim)
{

	reset();
	
	_volumes["posFileRasterized"] = std::make_unique<Volume>(resolution, TYPE_UCHAR);
	
	_volumes[CHANNEL_CONCETRATION] = std::make_unique<Volume>(_volumes[CHANNEL_MASK]->dim(), TYPE_DOUBLE);
	_volumes[CHANNEL_CONCETRATION]->clear();
	/*auto concetrationID = _volume->addChannel(
		_volume->getChannel(CHANNEL_MASK).dim(),
		TYPE_DOUBLE
	);*/
	//_volume->getChannel(CHANNEL_CONCETRATION).clear();


	size_t count = 0;
	{
		std::ifstream f(path);
		count = fast::getPosFileCount(f);
		f.close();
	}
	

	std::cout << count << " distributions in " << path << std::endl;
	if (count == 0) return false;

	index = index % count;	
	std::ifstream f(path);
	_sceneGeometry = fast::readPosFile(f, index, trim);
	f.close();
	
	if (_sceneGeometry.size() == 0)
		return false;

	rasterize(_sceneGeometry, *_volumes[CHANNEL_MASK]);

	
	/*
		Generate colord VBO for rendering
	*/
	const int colorN = 26;
	const uchar3 colors[colorN] = {		
		{ 0,117,220 }, //blue
		{ 43,206,72 }, //green
		{ 255,0,16 }, //red
		{ 240,163,255 },
		{ 153,63,0 },
		{ 76,0,92 },
		{ 25,25,25 },
		{ 0,92,49 },
		{ 255,204,153 },
		{ 128,128,128 },
		{ 148,255,181 },
		{ 143,124,0 },
		{ 157,204,0 },
		{ 194,0,136 },
		{ 0,51,128 },
		{ 255,164,5 },
		{ 255,168,187 },
		{ 66,102,0 },
		{ 94,241,242 },
		{ 0,153,143 },
		{ 224,255,102 },
		{ 116,10,255 },
		{ 153,0,0 },
		{ 255,255,128 },
		{ 255,255,0 },
		{ 255,80,5 }
	};

	for (auto obj : _sceneGeometry) {
		
		const auto & tmpGeom = obj->getTemplateGeometry();

		//Generate vbos of template particles
		auto vboIt = _geometryVBOs.find(tmpGeom);

		if (vboIt == _geometryVBOs.end()) {

			auto tm = std::dynamic_pointer_cast<fast::TriangleMesh>(tmpGeom);

			if (tm) {
				uchar3 color = colors[_geometryVBOs.size() % colorN];

				_geometryVBOs[tmpGeom] = getTriangleMeshVBO(
					*tm, vec4(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f, 1.0f)
				);
			}
		}
	}
	



	return true;
}

bool BatteryApp::loadFromMask(fast::Volume && mask)
{

	reset();

	_volumes[CHANNEL_MASK] = std::make_unique<fast::Volume>(std::move(mask));

	_volumes[CHANNEL_MASK]->binarize(1.0f);
	
	
	_volumes[CHANNEL_CONCETRATION] = std::make_unique<fast::Volume>(_volumes[CHANNEL_MASK]->dim(), TYPE_DOUBLE);
	_volumes[CHANNEL_CONCETRATION]->clear();
		
	_volumes[CHANNEL_MASK]->getPtr().createTexture();
	runAreaDensity();
	_volumeRaycaster->setVolume(*_volumes[CHANNEL_MASK]);

	return true;
}

void BatteryApp::reset()
{
	//_volume = make_unique<fast::Volume>();
	_volumes.clear();
	_volumeMC = std::move(VertexBuffer<VertexData>());	

	return;
	
}
