#include "RenderList.h"

void RenderList::clear() { _shaderToQueue.clear(); }

void RenderList::add(const std::shared_ptr<Shader> &shaderPtr,
                     RenderItem item) {
  _shaderToQueue[shaderPtr].push_back(item);
}

void RenderList::render() {

	glPushAttrib(GL_POLYGON_BIT);

  for (auto it : _shaderToQueue) {
    auto &shader = *it.first;
    const auto &queue = it.second;

    shader.bind();

    for (auto &item : queue) {

      for (auto &shaderOpt : item.shaderOptions) {

        mpark::visit(
            [&](auto arg) {
              auto &res = shader[shaderOpt.first];
              res = arg;
            },
            shaderOpt.second);
      }
	  	  
	  glCullFace(item.cullFace);
	  glPolygonMode(GL_FRONT_AND_BACK, item.polygonMode);

      item.vbo.render();
    }

    shader.unbind();
  }

  glPopAttrib();
}
