#pragma once
#include <args.h>

using uint = unsigned int;
class Module {
public:

	virtual ~Module() {}

	//Throw args::Error on error
	virtual void prepare() { }

	virtual void execute() { }	



};
