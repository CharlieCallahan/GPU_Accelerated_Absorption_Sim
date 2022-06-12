/* Copyright (c) 2021 Charlie Callahan
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef file_io_hpp
#define file_io_hpp

#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
namespace fio
{
    void save_to_bin_file(const char *data, uint32_t data_size, std::string filename); // saves to bin file
    void save_to_char_file(const char *data, uint32_t data_size, std::string filename);
    void append_to(std::string filename, const char *data);
    char *load_from_bin_file(std::string filename);
    void load_from_bin_file(std::string filename, char *mem_target, uint32_t size_bytes);
    void write_to(std::fstream *file, uint64_t bytePos, char *data, uint64_t sizeBytes); // overwrites sizeBytes bytes from data pointer to location bytePos through bytePos+sizeBytes in file.

    void create_empty_file_dense(std::string filename, uint64_t sizeBytes); // creates file and fills with 0s
}
#endif /* file_io_hpp */
