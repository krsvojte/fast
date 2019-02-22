#pragma once

#include "utility/mathtypes.h"

class Camera{

	public:

		Camera();
		
		static Camera defaultCamera(int width, int height);
		
		void setPosition(const vec3 &pos);
		void setLookat(const vec3 &dir);
		vec3 getLookat(){ return m_lookat; }
		void setUp(const vec3 &up);

		vec3 getPosition() const;
		vec3 getDirection() const;
		vec3 getUp() const;

		float getFarPlane() const;
		float getNearPlane() const;

		void setWindowDimensions(int width, int height);		
		void setAspectRatio(int width, int height);
		void setFov(float fov);
		
		void setPlanes(float znear, float zfar);
		float getNear() const{ return m_znear; }
		float getFar() const { return m_zfar; }
		float getFov() const { return m_fov; }

		mat4 getPerspective() const;
		mat4 getView() const;
		mat4 getPV() const;

		int getWindowWidth() const;
		int getWindowHeight() const;

		vec3 rayFromPixel(const vec2 & coord) const;
		vec2 project(const vec3 & pos) const;
		vec3 unproject(const vec2 & screen, float z = 0.0f) const;
		bool rayToScreen(const vec3 & rayOrigin, const vec3 & rayDir, vec2 * beginOut, vec2 * endOut) const;


		void rotateAroundLookat(float angleRad);

	protected:
		void updateV();
		virtual void updateP();

		mat4 m_P;
		mat4 m_V;

		vec3 m_pos;		
		vec3 m_up;
		vec3 m_lookat;

		float m_aspectRatio;
		float m_fov;
		float m_znear;
		float m_zfar;

		
		int m_windowWidth;
		int m_windowHeight;

};


