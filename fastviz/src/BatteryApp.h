#pragma once

#include "App.h"

#include "render/Camera.h"
#include "render/VertexBuffer.h"
#include "render/VolumeRaycaster.h"
#include "render/GLRenderer.h"

#include "utility/Options.h"

#include <fastlib/volume/Volume.h>

#include <fastlib/optimization/SimulatedAnnealing.h>
#include <fastlib/optimization/SDFPacking.h>

#include <fastlib/solver/DiffusionSolver.h>
#include <fastlib/solver/BICGSTAB.h>

#include <fastlib/geometry/Transform.h>
#include <fastlib/geometry/GeometryObject.h>




#include "Ui.h"

#include <memory>




struct Shader;


#define OPTIONS_FILENAME "../fastviz/options.json"
#define CHANNEL_MASK "000_mask"
#define CHANNEL_CONCETRATION "001_concetration"

class BatteryApp : public App {

public:
	BatteryApp();
		
	
protected:
	
	virtual void update(double dt) override;
	virtual void render(double dt) override;

	virtual void callbackMousePos(GLFWwindow * w, double x, double y) override;
	virtual void callbackMouseButton(GLFWwindow * w, int button, int action, int mods) override;
	virtual void callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods) override;
	virtual void callbackScroll(GLFWwindow * w, double xoffset, double yoffset) override;
	virtual void callbackChar(GLFWwindow * w, unsigned int code) override;
	

	virtual void reset();
	void runAreaDensity();

	bool loadFromFile(const std::string & folder);
	bool loadFromPosFile(const std::string & path, ivec3 resolution, size_t index, const fast::AABB & trim);
	bool loadFromMask(fast::Volume && mask);

	OptionSet _options;

	/*
		Render Settings
	*/
	Camera _camera;	
	ShaderDB _shaders;
	uint _currentRenderChannel;

	/*
		Renderers
	*/
	
	std::unique_ptr<VolumeRaycaster> _volumeRaycaster;

	/*
		Renderable objects
	*/
	Scene _scene;
	VertexBuffer<VertexData> _volumeMC;

	
	//std::unique_ptr<fast::Volume> _volume;
	std::map<std::string, std::unique_ptr<fast::Volume>> _volumes;
	
		
	
	std::unordered_map<
		std::shared_ptr<fast::Geometry>,
		VertexBuffer<VertexData>
	> _geometryVBOs;

	using SceneGeometry = std::vector<std::shared_ptr<fast::GeometryObject>>;

	SceneGeometry _sceneGeometry;

	VertexBuffer<VertexData> _aabbVBO;

	std::unique_ptr<fast::SDFPacking> _sdfpacking;
	bool _sdfpackingAutoUpdate;




	bool _autoUpdate;

	/*fast::SimulatedAnnealing<
		std::vector<fast::EigenTransform>
	> _sa;*/


	/*fast::SimulatedAnnealing<
		std::vector<fast::Ellipsoid>
	> _saEllipsoid;*/

	
	friend Ui;
	Ui _ui;	

	
	
	

private:
	

	
	

};