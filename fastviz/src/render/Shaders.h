#pragma once


#include <array>
#include <unordered_map>
#include <memory>
#include <string>

#include "Shader.h"

enum ShaderType  {
	SHADER_PHONG,
	SHADER_FLAT,
	SHADER_POSITION,
	SHADER_VOLUME_RAYCASTER,
	SHADER_VOLUME_SLICE,
	SHADER_COUNT
};


using ShaderDB = std::array<
	std::shared_ptr<Shader>, 
	ShaderType::SHADER_COUNT
>;


#define SHADER_PATH "../fastviz/src/shaders/"

const std::array<
	const char *,
	ShaderType::SHADER_COUNT
> g_shaderPaths = {
	"forwardphong",
	"flat",
	"position",
	"volumeraycast",
	"volumeslice"
};

/*
	Loads shaders into targetDB and returns any compilation error string encountered. Returns "" on success.
*/
std::string loadShaders(ShaderDB & targetDB);