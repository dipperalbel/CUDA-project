
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

/**
 * functions to load the picture on cpu
 */
std::vector<uint8_t> pgm_load(unsigned int &width, unsigned int &height, const std::string & file); // load pgm picture
void pgm_save(std::vector<uint8_t> & buffer, const unsigned int width, const unsigned int height, const std::string & file); // save pgm picture
std::vector<int16_t> lut_load_fixed(unsigned int &width, unsigned int &height, const std::string & file); // load lut

/**
 * functions to load the picture on gpu
 */
thrust::device_vector<uint8_t> pgm_load_gpu(unsigned int &width, unsigned int &height, const std::string & file); // load pgm picture on gpu memory
void pgm_save_gpu(thrust::device_vector<uint8_t> & buffer, const unsigned int width, const unsigned int height, const std::string & file); // save pgm picture on gpu memory
thrust::device_vector<int16_t> lut_load_fixed_gpu(unsigned int &width, unsigned int &height, const std::string & file); // load lut on gpu memory
