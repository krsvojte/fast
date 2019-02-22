#pragma once

//#include <variant>

#include <variant.hpp>

#include <GL/glew.h>
#include "utility/mathtypes.h"
#include <vector>



enum ShaderInterface {
	SHADER_INTERFACE_NONE = 0,
	SHADER_INTERFACE_UNIFORM = 1,
	SHADER_INTERFACE_ATTRIB = 2
};


using ShaderResourceValue = mpark::variant<
	vec2,vec3,vec4,mat2,mat3,mat4,double,float,int,bool
>;




struct ShaderResource {
	int location;
	int size;
	GLenum type;
	ShaderInterface shaderInterface;

	ShaderResource & operator= (vec2 val);
	ShaderResource & operator= (vec3 val);
	ShaderResource & operator= (vec4 val);

	ShaderResource & operator= (mat2 val);
	ShaderResource & operator= (mat3 val);
	ShaderResource & operator= (mat4 val);

	ShaderResource & operator= (double val);
	ShaderResource & operator= (float val);
	ShaderResource & operator= (int val);
	ShaderResource & operator= (bool val);

	ShaderResource & operator=(const std::vector<vec3> & arr);

	static ShaderResource none;
};