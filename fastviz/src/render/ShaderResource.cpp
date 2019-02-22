#include "ShaderResource.h"
#include <glm/gtc/type_ptr.hpp>
#include <fastlib/utility/GLGlobal.h>


ShaderResource & ShaderResource::operator=(const std::vector<vec3> & arr)
{
	if (location == -1) return *this;
	if (shaderInterface == SHADER_INTERFACE_UNIFORM) {
		GL(glUniform3fv(location, static_cast<int>(arr.size()), reinterpret_cast<const GLfloat *>(arr.data())));
	}
	else {
		assert(false);
	}

	return *this;
}

ShaderResource ShaderResource::none = { -1,0,0, SHADER_INTERFACE_NONE };



ShaderResource & ShaderResource::operator=(vec2 val)
{
	if (location == -1) return *this;
	assert(type == GL_FLOAT_VEC2);

	if (shaderInterface == SHADER_INTERFACE_UNIFORM) {
		GL(glUniform2fv(location, 1, glm::value_ptr(val)));
	}
	else {
		GL(glVertexAttrib2fv(location, glm::value_ptr(val)));
	}

	return *this;
}

ShaderResource & ShaderResource::operator=(vec3 val)
{
	if (location == -1) return *this;
	assert(type == GL_FLOAT_VEC3);

	if (shaderInterface == SHADER_INTERFACE_UNIFORM) {
		GL(glUniform3fv(location, 1, glm::value_ptr(val)));
	}
	else {
		GL(glVertexAttrib3fv(location, glm::value_ptr(val)));
	}

	return *this;
}

ShaderResource & ShaderResource::operator=(vec4 val)
{
	if (location == -1) return *this;
	assert(type == GL_FLOAT_VEC4);

	if (shaderInterface == SHADER_INTERFACE_UNIFORM) {
		GL(glUniform4fv(location, 1, glm::value_ptr(val)));
	}
	else {
		GL(glVertexAttrib4fv(location, glm::value_ptr(val)));
	}

	return *this;
}

ShaderResource & ShaderResource::operator=(mat2 val)
{
	if (location == -1) return *this;
	assert(type == GL_FLOAT_MAT2 && shaderInterface == SHADER_INTERFACE_UNIFORM);
	GL(glUniformMatrix2fv(location, 1, GL_FALSE, glm::value_ptr(val)));
	return *this;
}

ShaderResource & ShaderResource::operator=(mat3 val)
{
	if (location == -1) return *this;
	assert(type == GL_FLOAT_MAT3 && shaderInterface == SHADER_INTERFACE_UNIFORM);
	GL(glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(val)));
	return *this;
}

ShaderResource & ShaderResource::operator=(mat4 val)
{
	if (location == -1) return *this;
	assert(type == GL_FLOAT_MAT4 && shaderInterface == SHADER_INTERFACE_UNIFORM);
	GL(glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(val)));
	return *this;
}

ShaderResource & ShaderResource::operator=(double val)
{
	if (location == -1) return *this;
	assert(type == GL_DOUBLE);

	if (shaderInterface == SHADER_INTERFACE_UNIFORM){
		GL(glUniform1d(location, val));
	}
	else {
		GL(glVertexAttrib1d(location, val));
	}

	return *this;
}

ShaderResource & ShaderResource::operator=(float val)
{
	if (location == -1) return *this;
	assert(type == GL_FLOAT);

	if (shaderInterface == SHADER_INTERFACE_UNIFORM) {
		GL(glUniform1f(location, val));
	}
	else {
		GL(glVertexAttrib1f(location, val));
	}

	return *this;
}

ShaderResource & ShaderResource::operator=(int val)
{
	if (location == -1) return *this;
	//assert(type == GL_INT || type == GL_UNSIGNED_INT || type == GL_);

	if (shaderInterface == SHADER_INTERFACE_UNIFORM) {
		GL(glUniform1i(location, val));
	}
	else {
		GL(glVertexAttribI1i(location, val));
	}

	return *this;
}



ShaderResource & ShaderResource::operator=(bool val)
{
	if (location == -1) return *this;
	assert(type == GL_BOOL);

	if (shaderInterface == SHADER_INTERFACE_UNIFORM) {
		GL(glUniform1i(location, val));
	}
	else {
		GL(glVertexAttribI1i(location, val));
	}

	return *this;
}