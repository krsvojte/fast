#pragma once


#include "utility/mathtypes.h"

#include <GL/glew.h>
#include <vector>

struct Texture {



	Texture(GLenum type = GL_TEXTURE_2D, GLuint width = 0, GLuint height = 0, GLuint depth = 0);	

	~Texture();	

	/*
		Returns index of the texture unit
	*/
	int bindTo(GLenum textureUnit) const;	

	const GLuint ID() const { return _ID; }
	static void resetBindings(GLenum upto = GL_TEXTURE31 + 1);
	std::vector<unsigned char> toBuffer(int * comp, int *w, int *h) const;

	ivec3 size;
	GLenum type;

private:
	GLuint _ID = 0;
};

