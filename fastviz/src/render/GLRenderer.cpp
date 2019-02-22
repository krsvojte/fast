#include "GLRenderer.h"

#include <stack>

RenderList getSceneRenderList(const Scene &scene, const ShaderDB &shaders,
                              const Camera &camera) {
  RenderList rl;

  const ShaderType shaderType = SHADER_PHONG;
  const mat4 parentTransform = mat4(1.0f);

  for (auto it : scene.getObjects()) {
    const auto &obj = *it.second;

    RenderList::RenderItem item = {
        obj.getVBO(),
        obj.getShaderOptions(shaderType, camera, parentTransform)};

    rl.add(shaders[shaderType], item);
  }

  return rl;
}
