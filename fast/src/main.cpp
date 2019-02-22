#include "TauModule.h"
#include "AlphaModule.h"
#include "PackModule.h"
#include "SliceModule.h"

#include <iostream>
#include <args.h>
#include <memory>

using namespace std;

enum Action {
	NONE,
	TAU,
	ALPHA,
	PACK,
	SLICE
};

int main(int argc, char **argv) {

	args::ArgumentParser parser("Fast - Tortuosity and Reactive Area density calculator", "Vojtech Krs (2018-2019) vkrs@purdue.edu");

	args::HelpFlag help(parser, "Help", "Displays this message", { 'h', "help" });
	args::Positional<std::string> argAction(parser, "action", 
		"Specify action to perform:\n\t"
		"\"tau\" for Tortuosity\n\t"
		"\"alpha\" for Reactive Area Density \n\t"
		"\"pack\" for Packing\n\t"
		"\"slice\" for Slicing\n\t"
		"(Add -h or --help flag for help specific for the action)"
		, args::Options::Single
	);

	//

	if (argc == 1) {
		std::cout << parser;
		return 0;
	}

	try {
		
		parser.ParseArgs(std::vector<std::string>{argv[1]});
		Action action = NONE;
		if (args::get(argAction) == "tau") {
			action = TAU;
		}
		else if (args::get(argAction) == "alpha") {
			action = ALPHA;
		}
		else if (args::get(argAction) == "pack") {
			action = PACK;
		}
		else if (args::get(argAction) == "slice") {
			action = SLICE;
		}



		if (action != NONE) {
			
			parser.Reset();

			std::unique_ptr<args::Group> g;
			std::unique_ptr<Module> module;
			switch (action) {
			case TAU: 
				g = std::make_unique<args::Group>(parser, "Tortuosity", args::Group::Validators::DontCare);
				module = std::make_unique<TauModule>(*g);
				break;
			case ALPHA:
				g = std::make_unique<args::Group>(parser, "Reactive Area Density", args::Group::Validators::DontCare);
				module = std::make_unique<AlphaModule>(*g);
				break;
			case PACK:
				g = std::make_unique<args::Group>(parser, "Packing", args::Group::Validators::DontCare);
				module = std::make_unique<PackingModule>(*g);
				break;
			case SLICE:
				g = std::make_unique<args::Group>(parser, "Slice", args::Group::Validators::DontCare);
				module = std::make_unique<SliceModule>(*g);
				break;
			}
			

			try {
				parser.ParseCLI(argc, argv);
				module->prepare();								
			}
			catch (args::Help) {
				std::cout << parser;
				return 0;
			}
			catch (args::Error e) {
				std::cerr << e.what() << std::endl;
				std::cerr << "-------------------" << std::endl;
				std::cerr << parser;
				return 1;
			}


			try {
				module->execute();
			}
			catch (args::Error e) {
				std::cerr << e.what() << std::endl;				
				return 1;
			}


		}

		else {
			std::cerr << "Unknown action." << std::endl;
			std::cerr << parser;
			return 1;
		}

		//
	}
	catch (args::Help) {
		std::cout << parser;
		return 0;
	}
	catch (args::Error e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "-------------------" << std::endl;
		std::cerr << parser;
		return 1;
	}

	return 0;
}