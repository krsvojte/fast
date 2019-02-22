#include "BatteryApp.h"
#include <iostream>
#include <string>

int main(int argc, char ** argv){	

	try {
		BatteryApp app;

		while (app.run()) {
			/* ... */
		}

	
	}
	catch (const char * msg) {
		std::cerr << msg << std::endl;
		getchar();
		return -1;
	}
	catch (std::string  msg) {
		std::cerr << msg << std::endl;
		getchar();
		return -1;
	}

	return 0;
}