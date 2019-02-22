#include "utility/GLGlobal.h"

#include <GL/glew.h>
#include <iostream>

bool glewInitialized = false;

void logCerr(const char * label, const char * errtype){

	std::cerr << label << ": " << errtype << '\n';

}


#ifdef _DEBUG
bool GLError(const char *label /*= ""*/,
	const std::function<void(const char *label, const char *errtype)>
	&callback)
{
	bool hasErr = false;
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR)
	{	
		if (!callback) return false;
		switch (err) {
		case GL_INVALID_ENUM: callback(label,"GL_INVALID_ENUM"); break;
		case GL_INVALID_VALUE: callback(label, "GL_INVALID_VALUE"); break;
		case GL_INVALID_OPERATION: callback(label, "GL_INVALID_OPERATION"); break;
		case GL_STACK_OVERFLOW: callback(label, "GL_STACK_OVERFLOW"); break;
		case GL_STACK_UNDERFLOW: callback(label, "GL_STACK_UNDERFLOW"); break;
		case GL_OUT_OF_MEMORY: callback(label, "GL_OUT_OF_MEMORY"); break;
		case GL_INVALID_FRAMEBUFFER_OPERATION: callback(label, "GL_INVALID_FRAMEBUFFER_OPERATION"); break;
		default: callback(label, "Unknown Error"); break;
		}
		hasErr = true;
	}

	return hasErr;
}

#endif

bool resetGL()
{
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_TEXTURE_3D);
	

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);	
	glFrontFace(GL_CCW);
	
	
	return true;
}

FAST_EXPORT bool initGLEW()
{
	if (glewInitialized) return true;

	auto glewCode = glewInit();
	if (glewCode != GLEW_OK)
		throw (const char*)(glewGetErrorString(glewCode));

	glewInitialized = true;

	return true;
}


