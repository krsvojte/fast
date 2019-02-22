#include "VolumeRaycaster.h"
#include "render/PrimitivesVBO.h"

#include <fastlib/utility/GLGlobal.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#ifdef WIN32
	#define STBI_MSC_SECURE_CRT
#endif
#include "stb_image_write.h"

using namespace fast;


color3 jet(double v, double vmin, double vmax)
{
	color3 c = { 1.0,1.0,1.0 }; // white
	double dv;

	if (v < vmin)
		v = vmin;
	if (v > vmax)
		v = vmax;
	dv = vmax - vmin;

	if (v < (vmin + 0.25 * dv)) {
		c.r = 0;
		c.g = 4 * (v - vmin) / dv;
	}
	else if (v < (vmin + 0.5 * dv)) {
		c.r = 0;
		c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
	}
	else if (v < (vmin + 0.75 * dv)) {
		c.r = 4 * (v - vmin - 0.5 * dv) / dv;
		c.b = 0;
	}
	else {
		c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
		c.b = 0;
	}

	return(c);
}

std::vector<color4> transferJet() {

	std::vector<color4> arr(64);	
	for (auto i = 0; i < arr.size(); i++) {
		float t = i / float(arr.size() - 1);
		arr[i] = color4(jet(t, 0, 1),t*t * 0.05f);
	}

//	arr.front().w = 0.005f;

	return arr;

}

VolumeRaycaster::EnterExitVolume::EnterExitVolume()
	: 
	enterTexture(GL_TEXTURE_2D, 0, 0, 0),
	exitTexture(GL_TEXTURE_2D, 0, 0, 0)
{

}

void VolumeRaycaster::EnterExitVolume::resize(GLuint w, GLuint h) {
  if (w != enterTexture.size.x || h != enterTexture.size.y) {
    enterTexture.size = {w, h, 0};
    GL(glBindFramebuffer(GL_FRAMEBUFFER, enterFramebuffer.ID()));
    GL(glBindTexture(GL_TEXTURE_2D, enterTexture.ID()));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, enterTexture.ID(), 0));
    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  }
  if (w != exitTexture.size.x || h != exitTexture.size.y) {
    exitTexture.size = {w, h, 0};
    GL(glBindFramebuffer(GL_FRAMEBUFFER, exitFramebuffer.ID()));
    GL(glBindTexture(GL_TEXTURE_2D, exitTexture.ID()));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, NULL));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, exitTexture.ID(), 0));
    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  }
}



VolumeRaycaster::ScreenshotBuffer::ScreenshotBuffer()
	:
	tex(GL_TEXTURE_2D, 0, 0, 0)
{

}


void VolumeRaycaster::ScreenshotBuffer::resize(GLuint w, GLuint h)
{
	if (w != tex.size.x || h != tex.size.y) {
		tex.size = { w, h, 0 };
		GL(glBindFramebuffer(GL_FRAMEBUFFER, fbo.ID()));
		GL(glBindTexture(GL_TEXTURE_2D, tex.ID()));
		GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_FLOAT, NULL));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, tex.ID(), 0));
		GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	}
}

VolumeRaycaster::VolumeRaycaster(std::shared_ptr<Shader> shaderPosition,
                                 std::shared_ptr<Shader> shaderRaycast,
                                 std::shared_ptr<Shader> shaderSlice)
    : _shaderPosition(std::move(shaderPosition)),
      _shaderRaycast(std::move(shaderRaycast)),
      _shaderSlice(std::move(shaderSlice)), _cube(getCubeVBO()),
      _quad(getQuadVBO()), sliceMin({-1, -1, -1}), sliceMax({1, 1, 1}),      
	  _transferTexture(GL_TEXTURE_1D, 16, 1, 0),
	  _volTexture(0),
	  _volDim({0,0,0}),
	  showGradient(false),
	 _enableFiltering(true),
	_normalizeRange({0.0f,1.0f}),
	_screenshotMode(false)
	  {

  {
    			  
	setTransferJet();

  }
}


bool VolumeRaycaster::setVolume(const fast::Volume & volume)
{

	
	_volTexture = volume.getPtr().getGlID();
	_volDim = volume.dim();
	_volType = volume.type();
	enableFiltering(_enableFiltering);

	return true;
}

void VolumeRaycaster::render(const Camera &camera, ivec4 viewport) {
	GL(glEnable(GL_TEXTURE_3D));

  _enterExit.resize(viewport[2], viewport[3]);
 

  const ivec3 size = _volDim;

  // Render enter/exit texture
  {
    auto &shader = *_shaderPosition;

    mat4 M = mat4(1.0f);
    if (preserveAspectRatio) {

      // include bug for max?
      int maxDim = (size.x > size.y)
                       ? size.x
                       : size.y;
      maxDim =
          (maxDim > size.z) ? maxDim : size.z;

      vec3 scale =
          vec3(size) * (1.0f / static_cast<float>(maxDim));

      M = glm::scale(mat4(1.0f), scale);
    }

    shader.bind();
    shader["PVM"] = camera.getPV() * M;
    shader["minCrop"] = sliceMin;
    shader["maxCrop"] = sliceMax;

    GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.enterFramebuffer.ID()));
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(0, 0, viewport[2], viewport[3]);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    _cube.render();

    GL(glBindFramebuffer(GL_FRAMEBUFFER, _enterExit.exitFramebuffer.ID()));
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(0, 0, viewport[2], viewport[3]);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    _cube.render();

    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    shader.unbind();
  }

  // Raycast
  {
	 if (_screenshotMode) {
		 glBindFramebuffer(GL_FRAMEBUFFER, _screenshotBuffer.fbo.ID());
		 glClearColor(1, 1, 1, 1);
		 glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	 }

    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glDisable(GL_CULL_FACE);

    auto &shader = *_shaderRaycast;

    shader.bind();
    shader["transferFunc"] = _transferTexture.bindTo(GL_TEXTURE0);

	{
		GL(glActiveTexture(GL_TEXTURE1));
		GL(glBindTexture(GL_TEXTURE_3D, _volTexture));

		
		shader["volumeType"] = _volType;
		if (_volType == TYPE_DOUBLE || 
			_volType == TYPE_UINT //etc
			) {			
			shader["volumeTextureI"] = int(GL_TEXTURE1 - GL_TEXTURE0);
		}
		else {			
			shader["volumeTexture"] = int(GL_TEXTURE1 - GL_TEXTURE0);		
		}
		
		shader["resolution"] =
			vec3(1.0f / _volDim.x, 1.0f / _volDim.y,
				1.0f / _volDim.z);
	}		

	shader["normalizeLow"] = _normalizeRange.x; 
	shader["normalizeHigh"] = _normalizeRange.y;
	
    shader["enterVolumeTex"] = _enterExit.enterTexture.bindTo(GL_TEXTURE2);
    shader["exitVolumeTex"] = _enterExit.exitTexture.bindTo(GL_TEXTURE3);

    shader["steps"] = 128;
    shader["transferOpacity"] = 0.5f;
    shader["blackOpacity"] = opacityBlack;
    shader["whiteOpacity"] = opacityWhite;

	

    
    shader["showGradient"] = showGradient;

    shader["viewPos"] = camera.getPosition();

    _quad.render();

    shader.unbind();


	if (_screenshotMode) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
  }




}

void VolumeRaycaster::renderSlice(int axis, ivec2 screenPos,
                                  ivec2 screenSize) const {
  GL(glDisable(GL_CULL_FACE));
  GL(glViewport(screenPos.x, screenPos.y, screenSize.x, screenSize.y));
  auto &shader = *_shaderSlice;

  float t = sliceMin[axis];
  if (axis == 0) {
    t = sliceMin[2];
  } else if (axis == 1) {
    t = sliceMin[1];
  } else if (axis == 2) {
    t = sliceMin[0];
  }

  shader.bind();

  // Set uniforms
  {
	  GL(glActiveTexture(GL_TEXTURE0));
	  GL(glBindTexture(GL_TEXTURE_3D, _volTexture));
	  
	  if (_volType == TYPE_DOUBLE) {
		  shader["isDouble"] = true;
		  shader["texI"] = 0;
	  }
	  else {		  
		  shader["isDouble"] = false;
		  shader["tex"] = 0;
	  }
  }  
 

  shader["slice"] = (t + 1) / 2.0f;
  shader["axis"] = axis;

  // Render fullscreen quad
  _quad.render();

  shader.unbind();
}

void VolumeRaycaster::renderGrid(const Camera & camera, ivec4 viewport, Shader & shader, float opacity)
{	
	GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	GL(glViewport(0, 0, viewport[2], viewport[3]));


	glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);	
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GL(shader.bind());
	shader["useUniformColor"] = true;
	shader["uniformColor"] = vec4(0,0,0, opacity);
	shader["PV"] = camera.getPV();

	vec3 scaleVec = vec3(1.0f / _volDim.x, 1.0f / _volDim.y, 1.0f / _volDim.z);
	mat4 scale = glm::translate(mat4(1.0f), -vec3(1.0f) + scaleVec) * glm::scale(mat4(1.0f), scaleVec);

	for (auto x = 0; x < _volDim.x; x++) {
		for (auto y = 0; y < _volDim.y; y++) {
			for (auto z = 0; z < _volDim.z; z++) {				
				mat4 M = glm::translate(mat4(1.0f), vec3(x, y, z) * scaleVec * 2.0f) * scale;
				shader["M"] = M;
				shader["NM"] = M;
				GL(_cube.render());
			}
		}
	}

	

	GL(shader.unbind());
	

	glPopAttrib();

}


bool VolumeRaycaster::saveScreenshot(const Camera & camera, ivec4 viewport, const char * path)
{
	_screenshotMode = true;
	_screenshotBuffer.resize(viewport[2], viewport[3]);
	render(camera, viewport);
	_screenshotMode = false;


	const bool noTransparency = true;
	{
		
		uchar4 * data = nullptr;
		auto & tex = _screenshotBuffer.tex;
		
		data = new uchar4[tex.size.x * tex.size.y];
		GL(glBindTexture(GL_TEXTURE_2D, tex.ID()));
		GL(glGetTexImage(GL_TEXTURE_2D,0, GL_RGBA, GL_UNSIGNED_BYTE, data));
		GL(glBindTexture(GL_TEXTURE_2D, 0));		

		uchar4 * flipped = new uchar4[tex.size.x* tex.size.y];
		for (int x = 0; x < tex.size.x; x++) {
			for (int y = 0; y < tex.size.y; y++) {
			flipped[tex.size.x * y + x] = data[tex.size.x * (tex.size.y - y - 1) + x];
			if (noTransparency) {
				flipped[tex.size.x * y + x].w = 255;
			}
			}
		}

		int res = stbi_write_png(path, tex.size.x, tex.size.y, 4, flipped, tex.size.x * 4);
	

		delete[] flipped;

		return res != 1;
	}

}

void VolumeRaycaster::setTransferJet()
{
	auto transferVal = transferJet();

	//_transferTexture = Texture(GL_TEXTURE_1D, transferVal.size(), 1, 0);
	GL(glBindTexture(GL_TEXTURE_1D, _transferTexture.ID()));

	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));

	GL(glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA,
		static_cast<GLsizei>(transferVal.size()), 0, GL_RGBA, GL_FLOAT,
		transferVal.data()));
	GL(glBindTexture(GL_TEXTURE_1D, 0));
	_transferTexture.size.x = transferVal.size();
}

void VolumeRaycaster::setTransferGray()
{

	const std::vector<color4> transferVal = {
		{0,0,0,opacityBlack},
		{1,1,1,opacityWhite}
	};	

	//_transferTexture = Texture(GL_TEXTURE_1D, transferVal.size(), 1, 0);
	GL(glBindTexture(GL_TEXTURE_1D, _transferTexture.ID()));

	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));

	GL(glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA,
		static_cast<GLsizei>(transferVal.size()), 0, GL_RGBA, GL_FLOAT,
		transferVal.data()));
	GL(glBindTexture(GL_TEXTURE_1D, 0));
	_transferTexture.size.x = transferVal.size();
}

void VolumeRaycaster::enableFiltering(bool val)
{

	_enableFiltering = val;
	GL(glBindTexture(GL_TEXTURE_3D, _volTexture));

	if (_enableFiltering) {
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	}
	else {
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	}
	GL(glBindTexture(GL_TEXTURE_3D, 0));
}
