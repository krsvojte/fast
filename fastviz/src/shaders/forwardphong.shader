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


#define LIGHT_DIRECTIONAL 0
#define LIGHT_OMNI 1
#define LIGHT_SPOT 2

struct Light {
	vec4 pos;	
	vec3 color;
	float ambient;
	float coneAngle;
	vec3 coneDir;
	float attenuation;	
	bool castsShadow;
	float intensity;
};

int lightType(const in Light light){
	if(light.coneAngle != 0.0)
		return LIGHT_SPOT;
	if(light.pos.w == 0.0)
		return LIGHT_DIRECTIONAL;
	return LIGHT_OMNI;
}



struct Material {
	vec3 diffuse;
	vec3 ambient;
	vec3 specular;	
	float shininess;
	float opacity;
};


vec3 phong(
	const in Light light,
	const in Material mat,
	const in vec3 pos,
	const in vec3 viewDir, 
	const in vec3 normal	
){
	vec3 lightDir;
	float attenuation = 1.0;

	if(lightType(light) == LIGHT_DIRECTIONAL){
		lightDir = normalize(light.pos.xyz);		
	}
	else {
		vec3 toLight = light.pos.xyz - pos;
		float distToLight = length(toLight);
		lightDir = toLight / distToLight;
		attenuation = 1.0 / (1.0 + light.attenuation * distToLight * distToLight);

		if(lightType(light) == LIGHT_SPOT){
			float angle = degrees(acos(dot(-lightDir, normalize(light.coneDir))));
			if(angle > light.coneAngle){
				attenuation = 0.0;
			}
		}
	}

	
	
	//Ambient	
	vec3 ambient = mat.ambient * light.ambient * light.color;

	//Diffuse
	float diffuseMagnitude = max(dot(normal,lightDir),0.0);
	vec3 diffuse = clamp(mat.diffuse * diffuseMagnitude * light.color,0,1);	

	//Specular
	vec3 specular = vec3(0);	
	if (diffuseMagnitude > 0.0){ 
		float specularMagnitude = 
			pow(
				max(0.0, dot(reflect(-lightDir, normal),viewDir)), 
				mat.shininess
			);

		specular = mat.specular * specularMagnitude * light.color;
	}

	return ambient + attenuation*(diffuse + specular);
}


uniform vec3 viewPos;

void main(){

	Material mat;
	mat.diffuse = vec3(0.4) * fs_in.color.xyz;
	mat.ambient = vec3(0.1);
	mat.specular = vec3(0.1) * fs_in.color.xyz;
	mat.shininess = 0.8;

	Light light[2];
	light[0].pos = vec4(2,2,2,0.0);
	light[0].color = vec3(1,1,1);
	light[0].ambient = 1.0;
	

	light[1].pos = vec4(-2,2,-2,0.0);
	light[1].color = vec3(0.5);
	light[1].ambient = 1.0;
	
	vec3 N = normalize(fs_in.normal);

	vec3 viewDir =  normalize(fs_in.pos - viewPos);

	vec3 color = phong(light[0], mat, fs_in.pos, viewDir, N)
				+ phong(light[1], mat, fs_in.pos, viewDir, N);
	
	
	fragColor = vec4(color,fs_in.color.a);
}
