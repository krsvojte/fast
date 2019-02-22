#pragma once


class BatteryApp;
struct GLFWwindow;

/*
	Immediate mode GUI using dear-ImGui
*/
class Ui {

public:
	Ui(BatteryApp & app);

	void update(double dt);

	void callbackMouseButton(GLFWwindow * w, int button, int action, int mods);
	void callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods);
	void callbackScroll(GLFWwindow * w, double xoffset, double yoffset);
	void callbackChar(GLFWwindow * w,unsigned int code);

	bool isFocused() const;

private:
	BatteryApp & _app;

};
