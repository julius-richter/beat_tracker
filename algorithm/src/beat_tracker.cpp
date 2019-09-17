#include <iostream>
#include <stdexcept>
#include <string>
#include <stdlib.h>

// #include "utils.h"
// #include "dr_wav.cpp"
#include <kfr/all.hpp>


using namespace kfr;

int main(int argc, char** argv) 
{
	// check the number of parameters 
	if (argc < 2) {
		// Tell the user how to run the program
		std::cerr << "The algorithm need at least one parameter." << std::endl;
		return 1;
	}

	char* arg = argv[1];

	// convert parameter to int
	std::string parameter = "1";
	int parameter_int;
	try {
		std::size_t pos;
		parameter_int = std::stoi(parameter, &pos);
	} catch (std::invalid_argument const &ex) {
		std::cerr << "Invalid parameter: " << parameter << '\n';
	} catch (std::out_of_range const &ex) {
		std::cerr << "Number out of range: " << parameter << '\n';
	}
	
	// kfr::open_file_for_reading(argv[1]);


	// Open file as sequence of float`s, conversion is performed internally
	audio_reader_wav<float> reader(open_file_for_reading("/Users/juliusrichter/Documents/Uni/Masterarbeit/beat_tracker/data/audio/Hainsworth.SMC_001.wav"));
	univector2d<float> audio = reader.read_channels();

	// print(arg);

	return 0;
}