#pragma once
#include "Module.h"
#include <any.hpp>
#include <fstream>
#include <iostream>
#include <map>

using any = linb::any;
using RunRow = std::map<std::string, any>;

class CSVOutputModule : public Module {
public:

	CSVOutputModule(args::Group & parentGroup) :
		_argOutput(parentGroup, "output", "Output file", { 'o', "output" }, ""),
		_rowN(0)
	{

	}

	void overridePath(const std::string & path) {		
		_argOutput.Reset();
		args::EitherFlag f('o');		
		_argOutput.Match(f);
		_argOutput.ParseValue({ path });
		
		
	}

	void addGlobalRunRow(const RunRow & row) {
		_globalRow = std::make_unique<RunRow>(row);
	}

	
	void addRunRow(RunRow row) {

		
		if (_globalRow) {
			for (auto it : *_globalRow) {
				row[it.first] = it.second;
			}
		}



		bool isNew;
		auto & out = getOutputStream(&isNew);

		if (isNew) {			
			int cnt = 0;
			for (auto & item : row) {
				out << item.first;
				if (cnt + 1 != row.size())
					out << ",";
				cnt++;
			}
			out << "\n";
		}
				
		int cnt = 0;
		for (auto & item : row) {
			auto & a = item.second;
			if (a.type() == typeid(std::string))
				out << linb::any_cast<std::string>(a);
			else if (a.type() == typeid(float))
				out << linb::any_cast<float>(a);
			else if (a.type() == typeid(double))
				out << linb::any_cast<double>(a);
			else if (a.type() == typeid(uint))
				out << linb::any_cast<uint>(a);
			else if (a.type() == typeid(int))
				out << linb::any_cast<int>(a);
			else if (a.type() == typeid(bool))
				out << int(linb::any_cast<bool>(a));
			else
				out << "N/A type " << a.type().name();

			if (cnt + 1 != row.size())
				out << ",";
			cnt++;
		}
		out << "\n";

		out.flush();

		_rowN++;
		
	}

protected:

	std::ostream & getOutputStream(bool * isNewOut = nullptr) {
		if (_argOutput && isNewOut) {
			*isNewOut = true;
			std::ifstream f(_argOutput.Get());
			if (f.is_open())
				*isNewOut = false;
		}		
		else if (!_argOutput && isNewOut) {
			*isNewOut = (_rowN == 0);
		}
		

		if (_argOutput && !_fout.is_open()) {
			_fout.open(_argOutput.Get(), std::ios::app);
		}

		if (_fout.is_open())
			return _fout;

		return std::cout;
		
	}
	
	

	args::ValueFlag<std::string> _argOutput;

	

	std::ofstream _fout;
	int _rowN;

	std::unique_ptr<RunRow> _globalRow;

};