#pragma once

#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>


#include "render/ShaderResource.h"

using ShaderOptions = std::vector<
	std::pair<std::string, ShaderResourceValue>
>;

struct Shader {
	GLuint id;
	std::unordered_map<std::string, ShaderResource> resources;
	ShaderResource & operator [](const std::string & name);
	bool bind();
	static void unbind();
};

std::tuple<bool /*success*/, Shader /*shader*/, std::string /*error msg*/> 
compileShader(const std::string & code);




