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
#include "file_io.hpp"
void fio::save_to_bin_file(const char *data, uint32_t data_size, std::string filename)
{
    std::ofstream file(filename, std::ofstream::binary);
    if (file.is_open())
    {
        file.write((char *)&data_size, sizeof(uint32_t)); // write size of file to first 4 bytes
        file.write(data, data_size);
        file.close();
    }
    else
    {
        std::cout << "ERROR::FIO save_to_bin_file( " << filename << " ) Could not open file.\n";
    }
}
void fio::save_to_char_file(const char *data, uint32_t data_size, std::string filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file.write(data, data_size);
        file.close();
    }
    else
    {
        std::cout << "ERROR::FIO save_to_char_file( " << filename << " ) Could not open file.\n";
    }
}
void fio::append_to(std::string filename, const char *data)
{
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    if (file.is_open())
    {
        file << data;
        file.close();
    }
    else
    {
        std::cout << "ERROR:: append_to( " << filename << " ) Could not open file.\n";
    }
}
char *fio::load_from_bin_file(std::string filename)
{
    // Will return 0 if file fails to open
    std::ifstream file(filename, std::ios_base::in | std::ios_base::binary);
    uint32_t size;
    if (!file.is_open())
    {
        std::cout << "FIO::Error failed to open: " << filename << std::endl;
        return 0;
    }
    if (file.is_open())
    {
        file.read((char *)&size, sizeof(uint32_t));
        char *data = new char[size];
        file.read(data, size);
        return data;
    }
    file.close();
    return 0;
}
void fio::load_from_bin_file(std::string filename, char *mem_target, uint32_t size_bytes)
{
    std::ifstream file(filename, std::ios_base::in | std::ios_base::binary);
    uint32_t size;
    if (!file.is_open())
    {
        std::cout << "FIO::Error Failed to open: " << filename << std::endl;
        return;
    }
    if (file.is_open())
    {
        file.read((char *)&size, sizeof(uint32_t));
        // printf("file size: %i\n",size);
        if (size_bytes > size)
        {
            std::cout << "FIO::Warning requesting more bytes than contained in " << filename << std::endl;
            std::cout << "FIO::Requesting " << size_bytes << " bytes in file with " << size << " bytes" << std::endl;
            file.read(mem_target, size);
        }
        else
        {
            file.read(mem_target, size_bytes);
        }
    }
    file.close();
}
void fio::write_to(std::fstream *file, uint64_t bytePos, char *data, uint64_t sizeBytes)
{
    file->seekp(bytePos + 4);
    file->write(data, sizeBytes);
}

void fio::create_empty_file_dense(std::string filename, uint64_t sizeBytes)
{
    std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
    char filler = 0x1;
    uint32_t size32 = uint32_t(sizeBytes);
    file.write((char *)&(size32), sizeof(uint32_t));
    if (file.is_open())
    {
        for (uint64_t i = 0; i < sizeBytes; i++)
        {
            if (!file.write(&filler, 1))
            {
                std::cout << "ERROR::FIO create_empty_file_dense( " + filename + ") error writing to file, may be out of disk space \n";
            }
        }
        file.close();
    }
    else
    {
        std::cout << "ERROR::FIO create_empty_file_dense( " << filename << " ) Could not open file.\n";
    }
}
