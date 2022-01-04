
#pragma once

#include <cstdint>
#include <cstddef>

#include <thrust/device_vector.h>

/*
 * Census procedure
 * @outP - output picture on gpu
 * @inP - input picture on gpu
 * @width - picture width
 * @height - picture height
 * @stream - stream in which computations are performed
 */
void census( thrust::device_vector<uint8_t> & outP,
             const thrust::device_vector<uint8_t> & inP,
             unsigned int const width,
             unsigned int const height,
             cudaStream_t & stream );
