#version 440 core

////////////////////////////////////////
#pragma VERTEX

#include "passthrough.vert"


////////////////////////////////////////
#pragma FRAGMENT

#include "primitives.glsl"

in vec3 vposition;
out vec4 fragColor;

uniform sampler1D transferFunc;
uniform sampler3D volumeTexture;
uniform isampler3D volumeTextureI;

uniform sampler2D enterVolumeTex;
uniform sampler2D exitVolumeTex;

uniform int steps;
uniform float transferOpacity;

uniform vec3 minCrop;
uniform vec3 maxCrop;

uniform float whiteOpacity = 0.05;
uniform float blackOpacity = 0.001;

uniform vec3 viewPos;

const vec3 lightDir = vec3(1.0,0.0,1.0);
//const float voxelSize = 1 / 256.0;

uniform vec3 resolution;

uniform bool showGradient = false;
uniform int volumeType = TYPE_FLOAT;

uniform float normalizeLow = 0.0;
uniform float normalizeHigh = 0.0;


vec3 getGradient(vec3 pt){
	vec3 res = resolution*5;
	return vec3(
		(texture(volumeTexture, pt - vec3(res.x, 0.0, 0.0)).x - texture(volumeTexture, pt + vec3(res.x, 0.0, 0.0)).x) / res.x,
 		(texture(volumeTexture, pt - vec3(0.0, res.y, 0.0)).x - texture(volumeTexture, pt + vec3(0.0, res.y, 0.0)).x) / res.y,
 		(texture(volumeTexture, pt - vec3(0.0, 0.0, res.z)).x - texture(volumeTexture, pt + vec3(0.0, 0.0, res.z)).x) / res.z
 	); 	
}

vec3 getNormal(vec3 pt){
	return normalize(getGradient(pt));
}

float getLightMagnitude(vec3 pos, vec3 n, vec3 view){
	
	float ambient = 0.1;	
	float diff = max(dot(lightDir,n),0);
	float spec = 0.0;
	//if(diff > ambient){
		//vec3 r = reflect(-lightDir,n);
		//spec = max(dot(r,view),0.0);
		//spec = pow(spec,15.0);
	//}

	//return 0.01 * spec + 0.99*diff;
	return diff;

}

vec3 colormapJet(float v, float vmin, float vmax)
{
	vec3 c = vec3(1, 1, 1);
	float dv;

	if (v < vmin)
		v = vmin;
	if (v > vmax)
		v = vmax;
	dv = vmax - vmin;

	if (v < (vmin + 0.25f * dv)) {
		c.r = 0;
		c.g = 4 * (v - vmin) / dv;
	}
	else if (v < (vmin + 0.5f * dv)) {
		c.r = 0;
		c.b = 1 + 4 * (vmin + 0.25f * dv - v) / dv;
	}
	else if (v < (vmin + 0.75f * dv)) {
		c.r = 4 * (v - vmin - 0.5f * dv) / dv;
		c.b = 0;
	}
	else {
		c.g = 1 + 4 * (vmin + 0.75f * dv - v) / dv;
		c.b = 0;
	}

	return(c);
}

void main(){


	
	vec2 planePos = (vposition.xy + vec2(1)) * 0.5;
	vec3 enterPt = texture(enterVolumeTex,planePos).xyz;
	vec3 exitPt = texture(exitVolumeTex,planePos).xyz;		

	//Outside of bounding box
	if(enterPt == vec3(0.0f)) {				
		fragColor = vec4(0,0,0,0);
		return;
	}
	
	vec3 ray = -normalize(exitPt-enterPt);	
	float dt = 0.005;
	float N = distance(exitPt,enterPt) / dt;
	vec3 stepVec = ray*dt;

	vec3 pos = exitPt;
	fragColor = vec4(vec3(0.0),0.0);


	vec4 colorAcc = vec4(0);
	float alphaAcc = 0;
	
	for(float i=0; i < N; i+=1.0){

		
		vec4 color;

		if(volumeType == TYPE_DOUBLE){
			ivec2 val = texture(volumeTextureI,pos).rg;		
			uvec2 valu = uvec2(val.x, val.y);
			double dval = packDouble2x32(valu);		
			float volumeVal = float(dval);
			volumeVal = (volumeVal - normalizeLow) / (normalizeHigh - normalizeLow);
			color = texture(transferFunc,volumeVal);
		}
		else if(volumeType == TYPE_FLOAT || volumeType == TYPE_UCHAR ){
			float volumeVal = texture(volumeTexture,pos).r;		
			volumeVal = (volumeVal - normalizeLow) / (normalizeHigh - normalizeLow);
			color = texture(transferFunc,volumeVal);
		}
		else if(volumeType == TYPE_UCHAR4){
			color = texture(volumeTexture,pos).rgba;		
			
			vec3 blue = vec3(0,117/255,220/255);
			if(color.x == 0 && color.y == 0 && color.z == 0){
				color.a = 0;//blackOpacity;
				//color.xyz = vec3(1);
			}			
			/*else if(length(color.xyz - blue) > 0.9){
				color.a = 0.00;
			}*/
			else
				color.a = 0.25;
		}

		pos += stepVec;		

		vec3 gradient = getGradient(pos);
		float glen = length(gradient);


		//color.rgb = vec3(getLightMagnitude(pos, getNormal(pos), viewPos));
		if(showGradient){
			//color.xyz = colormapJet(glen, 0, 500);
			if(glen > 150.00){
				color.xyz = colormapJet(glen, 0, 250);
				color.xyz = vec3(glen/250);
				color.a *= 0.5;
			}
			else {
				color = vec4(vec3(0,0,0), dt*10);
			}
			//	color *= vec4(vec3(0,0,0), glen*0.0002);
			//else
			//	color *= vec4(vec3(1,1,1), dt*20);
		}


			//color.rgb *= (getLightMagnitude(pos, getNormal(pos), viewPos));

		

		
		
 
		//vec3 Cprev = colorAcc.xyz * colorAcc.a;
		//float Aprev = colorAcc.a;
		//vec3 Ci = color.xyz * color.a;
		//float Ai = color.a;

		//colorAcc.xyz =  (1 - Ai) * Ci + Cprev;
		//colorAcc.a = (1 - Ai) *  Ai + Aprev;

		//colorAcc.xyz = (1 - Aprev) * Ci + Cprev;
		//colorAcc.a = (1 - Aprev) * Ai + Aprev;

		//colorAcc = color*10;

		if(true){
			color.rgb *= color.a;
			//float alphaSample = 1.0 - pow((1.0 - color.a),dt);

			colorAcc =  (1 - colorAcc.a) * color +  colorAcc;

			//alphaAcc += alphaSample;
			//colorAcc.rgb  = color.rgb * 100;
			//colorAcc.a = 1;
		}




//		if(alphaAcc > 0.95)
			//break;

		//color= vec4(vec3(volumeVal),0.5);
		//color.rgb *= getLightMagnitude(pos,getNormal(pos),ray);
		//color.a *= 1.0;//transferOpacity;		
		//fragColor.rgb = mix(fragColor.rgb, color.rgb, color.a);

		//vec3 Cprev = fragColor.xyz ;
		//float Aprev = fragColor.a;
		//vec3 Ci = color.xyz ;
		//float Ai = color.a;

		//fragColor.xyz =  (1 - Ai) * Ci + Cprev;
		//fragColor.a = (1 - Ai) *  Ai + Aprev;

		//fragColor.xyz = (1 - Aprev) * Ci + Cprev;
		//fragColor.a = (1 - Aprev) * Ai + Aprev;


		//fragColor.rgb = (1 - color.a) * fragColor.rgb + color.rgb;
		//fragColor.a = (1 - color.a)

		//fragColor.a = mix(color.a,1.0,fragColor.a) ;	
		
		
	}

	vec3 clearColor = vec3(1.0);

	//colorAcc.rgb = mix(clearColor, colorAcc.rgb, colorAcc.a);

	fragColor = colorAcc;
	
}