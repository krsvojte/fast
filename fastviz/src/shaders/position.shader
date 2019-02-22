#version 440 core

////////////////////////////////////////
#pragma VERTEX


uniform vec3 minCrop;
uniform vec3 maxCrop;

in vec3 position;
out vec3 vposition;


uniform mat4 PVM;

void main()
{	   
	vec3 cropped = clamp(position,minCrop,maxCrop) ;
    gl_Position = PVM * vec4(cropped,1.0);
    vposition = (cropped + vec3(1.0))*0.5;
}



////////////////////////////////////////
#pragma FRAGMENT

in vec3 vposition;
out vec4 fragColor;

void main(){
	fragColor.xyz = vposition;		
	fragColor.a = 1;
}