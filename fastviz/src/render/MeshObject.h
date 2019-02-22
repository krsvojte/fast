#pragma once

#include "render/Scene.h"
#include "render/Shaders.h"

#include <fastlib/geometry/TriangleMesh.h>

class MeshObject : public SceneObject {

public:

	MeshObject() = default;
	~MeshObject() = default;

	MeshObject(fast::TriangleArray && mesh) : _mesh(mesh), SceneObject() {		
	}

	const fast::TriangleArray & getMesh() const;

	fast::TriangleArray getMesh();

	virtual ShaderOptions getShaderOptions(
		ShaderType shaderType,
		const Camera & cam, 
		mat4 parentTransform
	) const override;

protected:
	fast::TriangleArray _mesh;

	virtual bool _updateBuffer() const override;


};