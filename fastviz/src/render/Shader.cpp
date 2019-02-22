#include "Shader.h"

#include <regex>
#include <unordered_map>
#include <cassert>

#include <iostream>
#include <fastlib/utility/GLGlobal.h>




using namespace std;

const static unordered_map<GLenum, string> shaderEnumToString = {
	{ GL_VERTEX_SHADER, "VERTEX"},
	{ GL_FRAGMENT_SHADER, "FRAGMENT"},
	{ GL_GEOMETRY_SHADER, "GEOMETRY" },
	{ GL_TESS_CONTROL_SHADER, "TCS" },
	{ GL_TESS_EVALUATION_SHADER, "TES" }
};


unordered_map<GLenum, string> preprocessShaderCode(string code) {

	
	//std::replace(code.begin(), code.end(), '\r', ' ');
	//std::replace(code.begin(), code.end(), '\n', ' ');
	//std::cout << code << std::endl;

	const unordered_map<GLenum, regex> regexes = {
		{ GL_VERTEX_SHADER, regex("#pragma +VERTEX") },
		{ GL_FRAGMENT_SHADER, regex("#pragma +FRAGMENT") },
		{ GL_GEOMETRY_SHADER, regex("#pragma +GEOMETRY") },
		{ GL_TESS_CONTROL_SHADER, regex("#pragma +TCS") },
		{ GL_TESS_EVALUATION_SHADER, regex("#pragma +TES") },
	};

	//const regex pragmaRegex = regex("^.*#pragma(.*)$.$");
	//const regex toLF = regex(" \\r\\n | \\r(?!\\n) ");



	bool hasVertex = false;
	bool hasFragment = false;

	//Match pragmas delimiting individual shaders
	vector<pair<GLenum, smatch>> matches;
	for (auto it : regexes) {
		smatch match;
		if (regex_search(code, match, it.second)) {
			matches.push_back(make_pair(it.first, std::move(match)));
			if (it.first == GL_VERTEX_SHADER) hasVertex = true;
			if (it.first == GL_FRAGMENT_SHADER) hasFragment = true;
			//std::cout << "Matched " << it.first << std::endl;
		}
		
	}

	//Invalid if less than two shaders found, or missing vertex and fragment
	if (matches.size() < 2 || !hasFragment || !hasVertex)
		return unordered_map<GLenum,string>();  //return empty

	//Sort by order of occurence
	sort(matches.begin(), matches.end(), [](pair<GLenum, smatch> & a, pair<GLenum, smatch> & b) {
		return a.second.position() < b.second.position();
	});

	//Extract common part
	string common = code.substr(0, matches.front().second.position());
	

	//Extract individual shader sources
	unordered_map<GLenum, string> shaderSources;
	for (auto it = matches.begin(); it != matches.end(); it++) {
		auto next = it + 1;

		size_t begin = it->second.position() + it->second.length();		
		shaderSources[it->first] = common + code.substr(
			begin,
			(next == matches.end()) ? string::npos
									: (next->second.position() - begin)
		);		
	}

	return shaderSources;

}

string annotateLines(const string &str) {
	string newstr = str;
	size_t index = 0;
	int line = 1;
	while (true) {		
		index = newstr.find("\n", index);
		if (index == std::string::npos) break;		
		char buf[16];		
		sprintf(buf,"%d",line++);
		newstr.replace(index, 1, string("\n") + buf + string("\t"));
		
		index += strlen(buf) + 1 + 1;
	}

	return newstr;
	
}


std::tuple<bool /*success*/, Shader /*shader*/, string /*error msg*/>
compileShader(const string & code)
{
	
	auto shaderSources = preprocessShaderCode(code);
	if (shaderSources.size() == 0) {	
		return std::make_tuple(false, Shader(), string("Could not find vertex and fragment sub shaders.") );
	}

	/*for(auto & it : shaderSources){
		std::cout << it.second;
	}*/

		
	GLuint programID = glCreateProgram();	
	unordered_map<GLenum, GLint> shaderIDs;

	//Cleanup in case of failure
	auto failFun = [&](const std::string &err)  -> std::tuple<bool, Shader, string> {

		for (auto it : shaderIDs)
			glDeleteShader(it.second);
		shaderIDs.clear();

		glDeleteProgram(programID);
		programID = 0;

		return std::make_tuple(false,Shader(), err);		
	};

	/*
		Compile individual shaders
	*/
	string wholeShader = "";
	for (auto it : shaderSources) {
		GLuint id = glCreateShader(it.first);
		
		wholeShader.append(it.second);
		//Send source
		{
			GLint sourceLength = static_cast<GLint>(it.second.length());
			const GLchar * sourcePtr = (const GLchar *)it.second.c_str();
			glShaderSource(id, 1, &sourcePtr, &sourceLength);
		}

		glCompileShader(id);

		GLint isCompiled = 0;
		glGetShaderiv(id, GL_COMPILE_STATUS, &isCompiled);

		if (isCompiled == GL_FALSE) {
			GLint maxLength = 0;
			glGetShaderiv(id, GL_INFO_LOG_LENGTH, &maxLength);
			
			vector<char> buf(maxLength);
			glGetShaderInfoLog(id, maxLength, &maxLength, reinterpret_cast<GLchar *>(buf.data()));
					
						
			return failFun(
				annotateLines(it.second) + 
				shaderEnumToString.find(it.first)->second + "\n" + buf.data()
			);
		}


		glAttachShader(programID, id);
	}




	/*
		Link program
	*/
	glLinkProgram(programID);
	GLint isLinked = 0;
	glGetProgramiv(programID, GL_LINK_STATUS, (int *)&isLinked);
	if (isLinked == GL_FALSE)
	{
		GLint maxLength = 0;
		glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &maxLength);
		vector<char> buf(maxLength);
		glGetProgramInfoLog(programID, maxLength, &maxLength, reinterpret_cast<GLchar *>(buf.data()));
		
		
		return failFun(annotateLines(wholeShader) + "\n" + buf.data());
	}

	for (auto it : shaderIDs)
		glDeleteShader(it.second);


	/*
		Extract attribute and uniform locations
	*/
	auto getVars = [](GLint id, GLenum varEnum){
		const int MAX_VAR_LEN = 256;

		

		glUseProgram(id);
		//unordered_map<string, int> vars;
		unordered_map<string, ShaderResource> resources;

		int n;
		glGetProgramiv(id, varEnum, &n);
		char name[MAX_VAR_LEN];

		for (auto i = 0; i < n; i++) {
			ShaderResource sr;
			
			int nameLen;	
			if (varEnum == GL_ACTIVE_UNIFORMS)				
				glGetActiveUniform(id, i, MAX_VAR_LEN, &nameLen, &sr.size, &sr.type, name);
			else
				glGetActiveAttrib(id, i, MAX_VAR_LEN, &nameLen, &sr.size, &sr.type, name);		

			name[nameLen] = 0;
			if (nameLen > 3 && strcmp(name + nameLen - 3, "[0]") == 0)
				name[nameLen - 3] = 0;


			if (varEnum == GL_ACTIVE_UNIFORMS) {
				sr.location = glGetUniformLocation(id, name);
				sr.shaderInterface = ShaderInterface::SHADER_INTERFACE_UNIFORM;
			}
			else {
				sr.location = glGetAttribLocation(id, name);
				sr.shaderInterface = ShaderInterface::SHADER_INTERFACE_ATTRIB;
			}

			

			resources[name] = sr;
			
		}


		glUseProgram(0);
		return resources;
	};

	//auto uniforms = getUniforms(programID);
	//auto uniformBlocks = getVars(programID, GL_ACTIVE_UNIFORM_BLOCKS);	
	
	auto attribs = getVars(programID, GL_ACTIVE_ATTRIBUTES);
	auto uniforms = getVars(programID, GL_ACTIVE_UNIFORMS);
	attribs.insert(uniforms.begin(), uniforms.end());


	return std::make_tuple(true,
		Shader{programID,
		attribs}
	, "" );
	
}


ShaderResource & Shader::operator[](const std::string & name)
{
	auto it = resources.find(name);
	//Not found, return none type
	if (it == resources.end()) {
		return ShaderResource::none;
	}

	return it->second;
}


bool Shader::bind()
{
	GL(glUseProgram(id));	
	return true;
}

void Shader::unbind()
{	
	GL(glUseProgram(0));
}


