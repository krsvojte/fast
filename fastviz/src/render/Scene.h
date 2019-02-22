#pragma once

#include "render/RenderList.h"
#include "render/Shaders.h"

#include <map>


class Camera;


class SceneObject {

public:


	const VertexBuffer<VertexData> & getVBO() const;

	virtual ShaderOptions getShaderOptions(
		ShaderType shaderType,
		const Camera & cam,
		mat4 parentTransform
	) const = 0;


	mat4 getTransform() const { return _transform; }
	void setTransform(mat4 newTransform) { _transform = newTransform; }
	
protected:

	mat4 _transform;

	virtual bool _updateBuffer() const = 0;

	bool _isValid() const;
	void _invalidate();

	mutable bool _valid = false;
	mutable VertexBuffer<VertexData> _buffer;

};

/*
	Container for scene objects
*/
class Scene {

public:

	using Container = std::unordered_map<std::string, std::shared_ptr<SceneObject>>;

	std::shared_ptr<SceneObject> operator [](const std::string & name) const;

	/*
		Adds a named object to the scene
		If name already exists, a "_" character is appended to the name
	*/
	void addObject(std::string name, std::shared_ptr<SceneObject> object);

	/*
		Removes object from scene, returns false if object doesnt exist
	*/
	bool removeObject(const std::string & name);

	const Container & getObjects() const;


private:

	Container _objects;
};

/*


template<typename THost, typename TDevice>
struct HostDeviceResource{
		

private:

	virtual bool _synchronize() = 0;

	bool _validHost;
	bool _validDevice;
	THost _host;
	TDevice _device;
};

struct MeshVBO : public HostDeviceResource<fast::TriangleMesh, VertexData<VertexData>> {

	//implement syncrhonize
};
using VolumeTexture = HostDeviceResource<Volume, Texture>;


class SpatialObject {
	mat4 transform;
};

template <typename T>
class ResourceObject : public SpatialObject {	
	T resource;
};

class MeshObject : public ResourceObject<MeshVBO> {
};

using _Scene = std::map<std::string, SpatialObject>;

/ *
	Base class for scene object
	Owns cached gpu resource (vertex buffer)
	Non-const access to data invalides the gpu resource
* /

template <typename T>
class BaseObject {

	mat4 transform;

	const T & getResource() const;

protected:
	virtual bool _updateResource() const = 0;

	bool _isValid() const;
	void _invalidate();

	mutable bool _valid;
	mutable T _resource;
};

class VBOObject : public BaseObject<VertexBuffer<VertexData>> {
		
};

struct Texture;
class VolumeObject : public BaseObject<Texture> {

};



using Scene = std::map<
	std::string,
	std::shared_ptr<SceneObject>
>;
*/

