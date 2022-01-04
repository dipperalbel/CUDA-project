#pragma once

#include <cstdint>
#include <cstddef>

#include <thrust/device_vector.h>

/*
 * Interpolazione bilineare in fixed point
*/
void rectify( thrust::device_vector<uint8_t> & p,
              const thrust::device_vector<uint8_t> & pd,
              const thrust::device_vector<int16_t> & lut,
              unsigned int const width,
              unsigned int const height,
              unsigned int const src_width,
              cudaStream_t & stream );
