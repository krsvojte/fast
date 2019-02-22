#pragma once

#include "utility/mathtypes.h"

enum LightType {
	LIGHT_DIRECTIONAL = 0,
	LIGHT_OMNI = 1,
	LIGHT_SPOT = 2
};

struct Light {

	LightType type() const{	
		if (coneAngle != 0.0)
			return LIGHT_SPOT;
		if (pos.w == 0.0)
			return LIGHT_DIRECTIONAL;
		return LIGHT_OMNI;
	}
	vec4 pos; //dir if pos.w == 0.0, else omni/spot	
	color3 color; //all
	float ambient; //all	
	float coneAngle; //spot
	vec3 coneDir; //spot
	float attenuation; //omni, spot
	bool castsShadow = true;
	float intensity = 1.0f;
};

