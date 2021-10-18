
#ifndef file_io_hpp
#define file_io_hpp

#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
namespace fio{
    //double* load_bin_file(std::string filename); //imports binary file with
    void save_to_bin_file(const char * data, uint32_t data_size, std::string filename); //saves to bin file
    void save_to_char_file(const char * data, uint32_t data_size, std::string filename);
    void append_to(std::string filename, const char * data);
    char* load_from_bin_file(std::string filename);
    void load_from_bin_file(std::string filename, char* mem_target, uint32_t size_bytes);
    void write_to(std::fstream* file, uint64_t bytePos, char* data, uint64_t sizeBytes); //overwrites sizeBytes bytes from data pointer to location bytePos through bytePos+sizeBytes in file.
    
    void create_empty_file_dense(std::string filename, uint64_t sizeBytes); //creates file and fills with 0s
}
#endif /* file_io_hpp */
