#include "Texture.h"
#include <cassert>

#include <fastlib/utility/GLGlobal.h>

Texture::Texture(GLenum type /*= GL_TEXTURE_2D*/, GLuint width /*= 0*/, GLuint height /*= 0*/, GLuint depth /*= 0*/) : 
	type(type), 
	size{ width,height,depth }
{

	GL(glGenTextures(1, &_ID));
}

Texture::~Texture()
{
	GL(glDeleteTextures(1, &_ID));
}

int Texture::bindTo(GLenum textureUnit) const
{
	assert(textureUnit >= GL_TEXTURE0 && textureUnit < GL_TEXTURE31);
	GL(glActiveTexture(textureUnit));
	GL(glBindTexture(this->type, _ID));	
	GL(glEnable(this->type));

	return static_cast<int>(textureUnit) - GL_TEXTURE0;
}

void Texture::resetBindings(GLenum upto /*= GL_TEXTURE31*/)
{
	for (auto i = GL_TEXTURE0; i < static_cast<int>(upto); i++) {
		glActiveTexture(i);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindTexture(GL_TEXTURE_3D, 0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	}
}

std::vector<unsigned char> Texture::toBuffer(int * compOut, int *wOut, int *hOut) const
{

	//Only 2d supported at this moment
	assert(this->type == GL_TEXTURE_2D);


	glBindTexture(type, _ID);

	GLint format;
	int w, h;

	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);

	int comps;

	if (format == GL_DEPTH_COMPONENT || format == GL_R16F || format == GL_RED || format == GL_R) {
		comps = 1;
		format = GL_RED;
	}
	else if (format == GL_RG || format == GL_RG16F || format == GL_RG16) {
		comps = 2;
		format = GL_RG;
	}
	else if (format == GL_RGB || format == GL_RGB16F) {
		comps = 3;
		format = GL_RGB;
	}
	else if (format == GL_RGBA || format == GL_RGBA16F || format == GL_RGBA32F) {
		comps = 4;
		format = GL_RGBA;
	}

	size_t totalSize = w*h*comps;
	std::vector<unsigned char> buffer(totalSize);
	
	glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_BYTE, buffer.data());



	for (int y = 0; y < h / 2; y++) {
		for (int x = 0; x < w*comps; x++) {
			std::swap(buffer[y*w*comps + x], buffer[(h - 1 - y)*w*comps + x]);
		}
	}

	*compOut = comps;
	*wOut = w;
	*hOut = h;

	glBindTexture(GL_TEXTURE_2D, 0);
	return buffer;
}

