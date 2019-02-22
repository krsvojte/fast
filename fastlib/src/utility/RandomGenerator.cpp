#include "utility/RandomGenerator.h"

using namespace fast;

void fast::exec(int index, const std::function<void(void)> & f)
{
	f();
}


int fast::randomBi(RNGUniformInt & rnd) {
	return (rnd.next() % 2) * 2 - 1;
}