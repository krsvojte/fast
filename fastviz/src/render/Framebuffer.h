#pragma once

#include <fastlib/utility/GLGlobal.h>

struct FrameBuffer {
	FrameBuffer();
	~FrameBuffer();
	GLuint ID() const;

private:
	GLuint _ID;
};