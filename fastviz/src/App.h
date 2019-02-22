#pragma once

struct GLFWwindow;

#include "utility/mathtypes.h"
#include <deque>

class App {
	
	struct Window {
		int width = 0;
		int height = 0;
		GLFWwindow * handle = nullptr;
	};

	struct Input {
		bool mouseButtonPressed[8];
		vec2 mouseButtonPressPos[8];
		vec2 mousePos;
	};

public:
	App(const char * windowTitle);	
	bool run();
	void terminate();

	double getFPS() const;

	~App();

protected:	

	virtual void update(double dt);
	virtual void render(double dt);

	virtual void callbackMousePos(GLFWwindow * w, double x, double y);
	virtual void callbackMouseButton(GLFWwindow * w, int button, int action, int mods);
	virtual void callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods);
	virtual void callbackScroll(GLFWwindow * w, double xoffset, double yoffset);
	virtual void callbackResize(GLFWwindow * w, int width, int height);
	virtual void callbackChar(GLFWwindow * w, unsigned int code);


	Window _window;
	Input _input;

	double _lastTime;	
private:	
	bool _terminate;

	std::deque<double> _frameTimeHistory;
	

};