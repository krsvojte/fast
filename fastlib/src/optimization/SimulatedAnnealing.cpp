#include "optimization/SimulatedAnnealing.h"

#include <iostream>


float fast::defaultAcceptance(float e0, float e1, float temp)
{
	float p = 1.0f;
	if (e1 >= e0)
		p = exp(-(e1 - e0) / temp);	
	return p;
}

float fast::temperatureLinear(float fraction, size_t iteration)
{
	return (1.0f - fraction);
}

float fast::temperatureQuadratic(float fraction, size_t iteration)
{
	return (-fraction * fraction + 1);
}

float fast::temperatureExp(float fraction, size_t iteration)
{
	return exp(-fraction);
}



/*
float fast::defaultTemperature(float fraction) {
	return 1.0f - fraction;
}
*/
