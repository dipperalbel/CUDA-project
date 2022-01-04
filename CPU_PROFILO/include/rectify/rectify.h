#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

/*
 * Interpolazione bilineare in fixed point
*/
void rectify( std::vector<uint8_t> & p,
              const std::vector<uint8_t> & pd,
              const std::vector<int16_t> & lut,
              unsigned int const width,
              unsigned int const height,
              unsigned int const src_width);
