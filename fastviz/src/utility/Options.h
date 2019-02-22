#pragma once

#include "mathtypes.h"

#include <variant.hpp>


#include <string>
#include <vector>
#include <unordered_map>
#include <memory>


using OptionType = mpark::variant<
	std::string,
	char, int, 
	float, double, 
	bool, 
	vec2, vec3,	vec4,
	ivec2, ivec3, ivec4,
	mat2, mat3,	mat4	
>;

struct Option {
	Option(){}
	~Option(){}
	OptionType value;			
};





struct OptionSet {	
	std::unordered_map<std::string,Option> options;
	std::unordered_map<std::string,std::shared_ptr<OptionSet>> children;

	OptionSet & operator[](const std::string & childName);	
	const OptionSet & operator[](const std::string & childName) const;

	template <typename T>
	T & get(const std::string & optionName);


	template <typename T>
	void set(const std::string & optionName, T value);

	bool erase(const std::string & optionName);

	void clear();

};

template <typename T>
T & OptionSet::get(const std::string & optionName)
{
	auto & opt = options[optionName];

	//Construct if empty
	#if defined(NO_VARIANT)
		T & value = *static_cast<T*>(&opt.value);
		if(opt.type == OT_NONE){
			value = T();
		}

		return value;

	#else
		if (!mpark::holds_alternative<T>(opt.value))
			opt.value = T();	

		return mpark::get<T>(options[optionName].value);
	#endif
}

template <typename T>
void OptionSet::set(const std::string & optionName, T value)
{
	get<T>(optionName) = value;
}



std::ostream & operator << (std::ostream &, const std::shared_ptr<OptionSet> & opt);
std::istream & operator >> (std::istream &, std::shared_ptr<OptionSet> & opt);

std::ostream & operator << (std::ostream &, const OptionSet & opt);
std::istream & operator >> (std::istream &, OptionSet & opt);
