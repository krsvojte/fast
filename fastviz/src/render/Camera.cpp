#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <assert.h>
#include <iostream>


Camera::Camera(){
	m_P = glm::mat4(1.0f);
	m_V = glm::mat4(1.0f);
	m_lookat = glm::vec3(0.0f);
	m_pos = vec3(1, 0, 0);
	m_up = vec3(0, 1, 0);
	m_fov = 1.0f;
	m_znear = 0.001f;
	m_zfar = 1.0f;	
}


Camera Camera::defaultCamera(int width, int height)
{
	Camera c;
	c.setUp({0,1,0});
	c.setPosition(vec3( -5.0,2.5,-5 ) * 0.75f);
	c.setLookat({ 0,0,0 });
	c.setWindowDimensions(width,height);
	c.setFov({ 3.14f / 4.0f });
	c.setPlanes(0.0125f, 128.0f);
	return c;	
}

void Camera::setPosition(const glm::vec3 &pos){ 
	m_pos = pos; 
	updateV(); 
}
//void Camera::setDirection(const glm::vec3 &dir){ m_dir = dir; updateV(); }

void Camera::setUp(const glm::vec3 &up){ m_up = up; updateV(); }

glm::vec3 Camera::getPosition() const{ return m_pos; }
glm::vec3 Camera::getDirection() const{ return m_lookat - m_pos; }
glm::vec3 Camera::getUp() const{ return m_up; }

float Camera::getFarPlane() const { return m_zfar; }
float Camera::getNearPlane() const { return m_znear; }

void Camera::setWindowDimensions(int width, int height){
	m_windowWidth = width;
	m_windowHeight = height;
	setAspectRatio(m_windowWidth, m_windowHeight);
}


void Camera::setAspectRatio(int width, int height){
	m_aspectRatio = float(width) / float(height);
	updateP();
}
void Camera::setFov(float fov){
	m_fov = fov;
	updateP();
}

void Camera::setPlanes(float znear, float zfar){
	m_znear = znear;
	m_zfar = zfar;
	updateP();
}

glm::mat4 Camera::getPerspective() const{ return m_P; }
glm::mat4 Camera::getView() const { return m_V;}

glm::mat4 Camera::getPV() const{ return m_P * m_V; }

int Camera::getWindowWidth() const { return m_windowWidth; }
int Camera::getWindowHeight() const { return m_windowHeight; }

void Camera::updateV(){
	m_V = glm::lookAt(m_pos, m_lookat, m_up);	
	assert(!std::isnan(m_V[0][0]));
}

void Camera::updateP(){
	m_P = glm::perspective(m_fov, m_aspectRatio, m_znear, m_zfar);
	assert(!std::isnan(m_P[0][0]));
}


glm::vec3 Camera::rayFromPixel(const glm::vec2 & coord) const {

	glm::vec4 rayNorm = glm::vec4(coord.x / m_windowWidth, 1.0f - (coord.y / m_windowHeight), 0.0f, 0.0f) * 2.0f - glm::vec4(1.0f, 1.0f, 0.0f, 0.0f);
	rayNorm.z = -1.0f;
	rayNorm.w = 1.0f;

	glm::vec4 rayEye = glm::inverse(m_P)*rayNorm;
	rayEye.z = -1.0f;
	rayEye.w = 0.0f;

	glm::vec4 rayWorld = glm::inverse(m_V)*rayEye;

	glm::vec3 ray = glm::normalize(glm::vec3(rayWorld.x, rayWorld.y, rayWorld.z));

	return ray;

}


glm::vec2 Camera::project(const glm::vec3 & pos) const{


	glm::vec4 v = glm::vec4(pos, 1.0f);
	//Project
	v = m_P * m_V * v;

	

	//Viewport //Perspective division
	glm::vec2 s = glm::vec2(v.x / v.w, v.y / v.w);
	s.x = (s.x + 1.0f) / 2.0f;
	s.y = 1.0f - ((s.y + 1.0f) / 2.0f);

	s.x *= m_windowWidth;
	s.y *= m_windowHeight;
	

	return s;
}


glm::vec3 Camera::unproject(const glm::vec2 & screen, float z) const{

	glm::vec4 Vp = glm::vec4(0, 0, m_windowWidth, m_windowHeight);
	return glm::unProject(glm::vec3(screen, z), m_V, m_P, Vp);
}


bool Camera::rayToScreen(const glm::vec3 & rayOrigin, const glm::vec3 & rayDir, glm::vec2 * beginOut, glm::vec2 * endOut) const{

	const float maxT = m_zfar;
	glm::vec2 begin = project(rayOrigin);
	glm::vec2 end = project(rayOrigin + rayDir * m_zfar);

	glm::vec2 bounds[2];
	bounds[0] = glm::vec2(0.0f);
	bounds[1] = glm::vec2(m_windowWidth, m_windowHeight);
	begin = glm::clamp(begin, bounds[0], bounds[1]);
	end = glm::clamp(end, bounds[0], bounds[1]);

	//Out of screen
	if (begin.x == end.x && begin.y == end.y) return false; 

	*beginOut = begin;
	*endOut = end;

	return true;
}


void Camera::rotateAroundLookat(float angleRad)
{
	vec3 dpos = m_pos - m_lookat;

	mat4 T = glm::translate(mat4(1.0f), -dpos) * glm::rotate(mat4(1.0f), angleRad, m_up) * glm::translate(mat4(1.0f), dpos);

	setPosition(vec3(T*vec4(m_pos, 1)));

}

void Camera::setLookat(const glm::vec3 &dir)
{
	m_lookat = dir;
	updateV();
}
