
#include <preprocessor/pgm.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/copy.h>

/*
 * I/O ROUTINES
 */

/**
 * Skips the comment line in pgm picture
 */
static void pnm_skip_comments(std::istream &i)
{
    while (isspace(i.peek()))
    {
        while (isspace(i.peek()))
        {
            i.get();
        }
        if (i.peek() == '#')
        {
            while (i.peek() != '\r' && i.peek() != '\n')
            {
                i.get();
            }
        }
    }
}

/**
 * Reads header of pgm picture and assigns width and height
 */
static char pnm_read_header(unsigned int &width, unsigned int &height, std::istream &iss)
{
    char h = 0, t = 0;
    int max_val = 0;

    // check pnm header
    h = iss.get();
    t = iss.get();
    if (!(h == 'P' && (t == '5' || t == '6')))
    {
        return '\0';
    }

    pnm_skip_comments(iss);
    iss >> width;
    pnm_skip_comments(iss);
    iss >> height;
    pnm_skip_comments(iss);
    iss >> max_val;
    iss.get();  // TODO: use a getline fn

    return t;
}

/*
 * Load a PGM image, reserve memory with new and return geometry
 *
 * @param [out] width|height geometry of image
 * @param [in] file filename
 * @return a pointer to image first pixel allocated with new operator,
 * @return or NULL in case of error
*/
std::vector<uint8_t> pgm_load(unsigned int &width, unsigned int &height, const std::string & file)
{
    std::ifstream istr(file, std::ios::in | std::ios::binary);

    if (pnm_read_header(width, height, istr) != '5')
    {
        throw std::runtime_error{ "pgm_load failed" };
    }

    std::vector<uint8_t> buffer(width * height);

    std::vector<char> parsedData{ std::istreambuf_iterator<char>(istr), std::istreambuf_iterator<char>() };
    std::copy(std::begin(parsedData), std::end(parsedData), reinterpret_cast<char*>(buffer.data()));

    return buffer;
}


void pgm_save(std::vector<uint8_t> & buffer, const unsigned int width, const unsigned int height, const std::string & file)
{
    std::ofstream ostr(file, std::ios::out | std::ios::binary);

    ostr << "P5\n" << width << ' ' << height << "\n255\n";

    ostr.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    if (!ostr)
    {
        throw std::runtime_error{ "pgm_save failed" };
    }
}

std::vector<int16_t> lut_load_fixed(unsigned int &width, unsigned int &height, const std::string & file)
{
    std::vector<int16_t> buffer(2 * width * height);
    std::ifstream istr(file, std::ios::in | std::ios::binary);
    std::vector<char> parsedData{ std::istreambuf_iterator<char>(istr), std::istreambuf_iterator<char>() };

    std::copy(std::begin(parsedData), std::end(parsedData), reinterpret_cast<char*>(buffer.data()));

    return buffer;
}
