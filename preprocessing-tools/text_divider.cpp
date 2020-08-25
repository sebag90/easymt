// Takes as input one txt file with alligned sentences separated by \t
// divides the languages and produces 2 files: filename_output.l1 and filename_output.l2

// ex. This is the first sentence	This is the translation of the first sentence

// SYNTAX: [filename input file] [filename_output]


// for older g++ version use -lstdc++fs flag to 
// compile and use <experimental/filesystem> instead of <filesystem>

#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]){
	if (argc == 3){
		std::ifstream file;
		file.open(argv[1]);
		std::string temp;
		
		std::string outputname(argv[2]);
		
		std::ofstream file_one (outputname+ ".l1");
		std::ofstream file_two (outputname + ".l2");



		while(getline(file, temp)){
		    std::size_t found = temp.find("\t");
		    file_one <<  temp.substr(0,found) + "\n";
		    file_two << temp.substr(found+1) + "\n";
		}
		
		file_one.close();
		file_two.close();
		file.close();
	}
	else{
	std::cout << "missing or invalid argument: \n" 
			  << "main [filename input file] [filename output]"	<< std::endl;
	}
}
