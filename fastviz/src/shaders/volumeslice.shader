#version 440 core

////////////////////////////////////////
#pragma VERTEX

#include "passthrough.vert"


////////////////////////////////////////
#pragma FRAGMENT

in vec3 vposition;
uniform sampler3D tex;
uniform isampler3D texI;

uniform float slice;
//uniform mat3 R;
uniform int axis;
uniform bool isDouble = false;

out vec4 fragColor;

void main(){

	vec2 coord = (vposition.xy + vec2(1)) * 0.5;	



	vec3 coord3D;
	if(axis == 0){
		coord.x = 1.0 - coord.x;
		coord3D = vec3(coord, slice);
	}
	else if(axis == 1){
		coord3D = vec3(1.0 - coord.x, slice, 1.0 - coord.y);
	}
	else if(axis == 2)	
		coord3D = vec3(slice, coord.x, coord.y);

	//vec3 coord3D = R * vec3(coord,slice);
	


	float volumeVal = 0;

	if(isDouble){
		ivec2 val = texture(texI,coord3D).rg;		
		uvec2 valu = uvec2(val.x, val.y);
		double dval = packDouble2x32(valu);		
		volumeVal = float(dval);
	}
	else {
		 volumeVal = texture(tex,coord3D).r;		
	}


	fragColor.xyz = vec3(volumeVal);		
	fragColor.a = 1;
}