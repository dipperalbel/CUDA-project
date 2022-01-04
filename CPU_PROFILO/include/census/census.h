
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

/*
 * Census procedure
 * @outP - output picture on gpu
 * @inP - input picture on gpu
 * @width - picture width
 * @height - picture height
 * @stream - stream in which computations are performed
 */
void census( std::vector<uint8_t> & outP,
             const std::vector<uint8_t> & inP,
             unsigned int const width,
             unsigned int const height);
