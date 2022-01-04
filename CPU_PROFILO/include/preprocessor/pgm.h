
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

/**
 * functions to load the picture on cpu
 */
std::vector<uint8_t> pgm_load(unsigned int &width, unsigned int &height, const std::string & file); // load pgm picture
void pgm_save(std::vector<uint8_t> & buffer, const unsigned int width, const unsigned int height, const std::string & file); // save pgm picture
std::vector<int16_t> lut_load_fixed(unsigned int &width, unsigned int &height, const std::string & file); // load lut
