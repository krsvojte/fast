#version 430 

#pragma VERTEX
#include "default.vertex"


#pragma FRAGMENT
layout(location = 0) out vec4 fragColor;

in VARYING {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	vec4 color;
} fs_in;


uniform vec4 uniformColor;
uniform bool useUniformColor = false;

void main(){
	
	if(useUniformColor)
		fragColor = uniformColor;
	else		
		fragColor = fs_in.color;
}
