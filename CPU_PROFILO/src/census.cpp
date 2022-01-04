
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <census/census.h>

#define offset 2

void census( std::vector<uint8_t> & outPs,
             const std::vector<uint8_t> & inPs,
             unsigned int const width,
             unsigned int const height)
{
    for (int i{}; i < (int)width; ++i)
    {
        for (int j{}; j < (int)height; ++j)
        {
            const int globalIdx = j * width + i;

            int greaterPixelsCount = 0;
            const uint8_t inP = inPs[globalIdx];
    
            const int endIdxX = i + offset;
            const int endIdxY = j + offset;
            for ( int idxY{ j - offset }; idxY <= endIdxY; ++idxY )
            {
                for ( int idxX{ i - offset }; idxX <= endIdxX; ++idxX )
                {
                  if ( idxX < width && idxY < height )
                  {
                    const int closeIdx = idxY * width + idxX;
                    const uint8_t closeValue = inPs[closeIdx];
                    if (closeValue > inP)
                    {
                        ++greaterPixelsCount;
                    }
                  }
                }
            }
    
            outPs[globalIdx] = greaterPixelsCount;
        }
    }
}
