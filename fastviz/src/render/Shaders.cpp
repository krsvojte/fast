#include "Shaders.h"

#include <string>
#include "utility/IOUtility.h"


using namespace std;

std::string loadShaders(ShaderDB & targetDB)
{	
	std::string errString;
	for (auto i = 0; i < ShaderType::SHADER_COUNT; i++) {

		const auto path = SHADER_PATH + string(g_shaderPaths[i]) + ".shader";
		auto src = readFileWithIncludes(path);

		if (src.length() == 0)
			throw "Failed to read " + path;

		if (targetDB[i] == nullptr) {
			targetDB[i] = make_shared<Shader>();
		}

//		std::tuple<bool, Shader, string /*error msg*/>
		auto ret = compileShader(src);
		bool ok = std::get<0>(ret);
		if (ok)
			*targetDB[i] = std::get<1>(ret);
		else {			
			errString.append(path + ":\n");			
			errString.append(std::get<2>(ret));				
			errString.append("\n");			
		}
	}

	return errString;
}
