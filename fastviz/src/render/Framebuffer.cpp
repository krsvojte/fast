#include "Framebuffer.h"


FrameBuffer::FrameBuffer()
{
	glGenFramebuffers(1, &_ID);
}

FrameBuffer::~FrameBuffer()
{
	glDeleteFramebuffers(1, &_ID);
}

GLuint FrameBuffer::ID() const
{
	return _ID;
}

