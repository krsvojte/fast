#pragma once

#include "render/Scene.h"
#include "render/RenderList.h"
#include <unordered_map>


RenderList getSceneRenderList(
	const Scene & scene,
	const ShaderDB & shaders,
	const Camera & camera
);


