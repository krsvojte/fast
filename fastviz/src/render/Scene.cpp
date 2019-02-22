#include "Scene.h"


bool SceneObject::_isValid() const
{
	return _valid;
}

const VertexBuffer<VertexData> & SceneObject::getVBO() const
{
	if (!_isValid()) {
		_valid = _updateBuffer();
		if (!_valid)
			throw "Failed to update gpu buffer";
	}
	return _buffer;
}

void SceneObject::_invalidate()
{
	_valid = false;
}

////////////////////////

std::shared_ptr<SceneObject> Scene::operator[](const std::string & name) const
{
	auto it = _objects.find(name);
	if (it == _objects.end())
		return nullptr;
	return it->second;
}

void Scene::addObject(std::string name, std::shared_ptr<SceneObject> object)
{

	auto it = _objects.find(name);
	while (it != _objects.end()) {
		name.append("_");
		it = _objects.find(name);
	}

	_objects[name] = std::move(object);

}

bool Scene::removeObject(const std::string & name)
{
	auto it = _objects.find(name);
	if (it == _objects.end())
		return false;

	_objects.erase(it);

	return true;
}

const Scene::Container & Scene::getObjects() const
{
	return _objects;
}
