#pragma once

#include "render/VertexBuffer.h"
#include "render/Shader.h"


struct RenderList {

	struct RenderItem {
		const VertexBuffer<VertexData> & vbo;
		const ShaderOptions shaderOptions;
		const GLenum polygonMode = GL_FILL; //GL_POINT | GL_LINE | GL_FILL
		const GLenum cullFace = GL_BACK; // GL_BACK | GL_FRONT | GL_FRONT_AND_BACK
	};

	void clear();

	void add(
		const std::shared_ptr<Shader> & shaderPtr,
		RenderItem item
	);

	void render();



private:
	std::unordered_map<
		std::shared_ptr<Shader>,
		std::vector<RenderItem>
	> _shaderToQueue;
};