
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <census/census.h>

#define threadsPerBlockX 64
#define threadsPerBlockY 8
#define offset 2
#define smx (threadsPerBlockX + 2 * offset)
#define smy (threadsPerBlockY + 2 * offset)
#define smSize ( smx * smy )

__global__ void census_cuda( uint8_t * outP, uint8_t const * inP, int const width, int const height );

void census( thrust::device_vector<uint8_t> & outP,
             const thrust::device_vector<uint8_t> & inP,
             unsigned int const width,
             unsigned int const height,
             cudaStream_t & stream )
{
    dim3 blockSize{ threadsPerBlockX, threadsPerBlockY };
    dim3 gridSize{ (width + threadsPerBlockX - 1) / threadsPerBlockX, (height + threadsPerBlockY - 1) / threadsPerBlockY }; 
    census_cuda<<<gridSize, blockSize, 0, stream>>>(outP.data().get(), inP.data().get(), (int)width, (int)height);
}

__global__ void census_cuda( uint8_t * outP, uint8_t const * inP, int const width, int const height )
{
    const int ti = threadIdx.x;
    const int ai = ti + offset;
  
    const int tj = threadIdx.y;
    const int aj = tj + offset;
  
    const int i  = ti + blockDim.x * blockIdx.x;
    const int j  = tj + blockDim.y * blockIdx.y;

    const int sharedIdx = aj * smx + ai;
    const int globalIdx = j * width + i;

    __shared__ uint8_t inPixels[smSize];

    if (i < width && j < height)
    {
        if (ti < offset)
        {
            if (i >= offset)
            {
                inPixels[sharedIdx - offset] = inP[globalIdx - offset];
            }
            else
            {
                inPixels[sharedIdx - offset] = 0;
            }
        }

        if (tj < offset)
        {
            if (j >= offset)
            {
                inPixels[(aj - offset) * smx + ai] = inP[(j - offset) * width + i];
            }
            else
            {
                inPixels[(aj - offset) * smx + ai] = 0;
            }
        }

        inPixels[sharedIdx] = inP[globalIdx];

        if (ti >= blockDim.x - offset)
        {
            if (i + offset < width)
            {
                inPixels[sharedIdx + offset] = inP[globalIdx + offset];
            }
            else
            {
                inPixels[sharedIdx + offset] = 0;
            }
        }

        if (tj >= blockDim.y - offset)
        {
            if (j + offset < height)
            {
                inPixels[(aj + offset) * smx + ai] = inP[(j + offset) * width + i];
            }
            else
            {
                inPixels[(aj + offset) * smx + ai] = 0;
            }
        }

        __syncthreads();

        int greaterPixelsCount = 0;
        const uint8_t inP = inPixels[sharedIdx];

        const int endIdxX = ai + offset;
        const int endIdxY = aj + offset;
        #pragma unroll 5
        for ( int idxY{ aj - offset }; idxY <= endIdxY; ++idxY )
        {
            #pragma unroll 5
            for ( int idxX{ ai - offset }; idxX <= endIdxX; ++idxX )
            {
                const int closeIdx = idxY * smx + idxX;
                const uint8_t closeValue = inPixels[closeIdx];
                if (closeValue > inP)
                {
                    ++greaterPixelsCount;
                }
            }
        }

        outP[globalIdx] = greaterPixelsCount;
    }
}
