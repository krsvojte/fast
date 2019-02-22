#pragma once
#include "render/Texture.h"
#include "render/Camera.h"
#include "render/Shader.h"
#include "render/VertexBuffer.h"
#include "render/Framebuffer.h"

#include <fastlib/volume/Volume.h>



struct VolumeRaycaster {

	struct EnterExitVolume {
		EnterExitVolume();
		void resize(GLuint w, GLuint h);

		FrameBuffer enterFramebuffer;
		FrameBuffer exitFramebuffer;
		Texture enterTexture;
		Texture exitTexture;
	};

	struct ScreenshotBuffer {
		ScreenshotBuffer();
		void resize(GLuint w, GLuint h);
		FrameBuffer fbo;
		Texture tex;

	};

	VolumeRaycaster(
		std::shared_ptr<Shader> shaderPosition,
		std::shared_ptr<Shader> shaderRaycast,
		std::shared_ptr<Shader> shaderSlice
	);

	
	bool setVolume(const fast::Volume & volume);

	void render(
		const Camera & camera,
		ivec4 viewport
	);

	void renderSlice(int axis, ivec2 screenPos, ivec2 screenSize) const;

	void renderGrid(const Camera & camera, ivec4 viewport, Shader & shader, float opacity = 0.1f);

	bool saveScreenshot(
		const Camera & camera,
		ivec4 viewport,
		const char * path);

	vec3 sliceMin;
	vec3 sliceMax;

	float opacityWhite;
	float opacityBlack;

	bool preserveAspectRatio;
	bool showGradient;

	void setNormalizeRange(vec2 range) {
		_normalizeRange = range;
	}
	

	void setTransferJet();
	void setTransferGray();

	void enableFiltering(bool val);

private:
	EnterExitVolume _enterExit;


	ScreenshotBuffer _screenshotBuffer;

	
	Texture _transferTexture;

	GLuint _volTexture;
	ivec3 _volDim;
	PrimitiveType _volType;
	bool _enableFiltering;
	vec2 _normalizeRange;
	

	VertexBuffer<VertexData> _cube;
	VertexBuffer<VertexData> _quad;

	std::shared_ptr<Shader> _shaderPosition;
	std::shared_ptr<Shader> _shaderRaycast;
	std::shared_ptr<Shader> _shaderSlice;

	bool _screenshotMode;

};