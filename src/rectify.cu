#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <rectify/rectify.h>

#define threadsPerBlock 256
int const num_threads = 65536;

int const fixed = 2;

__global__ void rectify_cuda( uint8_t *const p,
                              uint8_t const *const pd,
                              int16_t const *const lut,
                              unsigned int const width,
                              unsigned int const height,
                              unsigned int const src_width);

/*
 * Interpolazione bilineare in fixed point
*/
void rectify( thrust::device_vector<uint8_t> & p,
              const thrust::device_vector<uint8_t> & pd,
              const thrust::device_vector<int16_t> & lut,
              unsigned int const width,
              unsigned int const height,
              unsigned int const src_width,
              cudaStream_t & stream )
{
	int num_thread_blocks = (num_threads + threadsPerBlock - 1) / threadsPerBlock;
    rectify_cuda<<<num_thread_blocks, threadsPerBlock, 0, stream>>>( p.data().get(), pd.data().get(), lut.data().get(), width, height, src_width );
}

/*
 * Modifica 03.13 FG
*/
__global__ void rectify_cuda( uint8_t *const p,
                              uint8_t const *const pd,
                              int16_t const *const lut,
                              unsigned int const width,
                              unsigned int const height,
                              unsigned int const src_width )
{
    int block_size = blockDim.x * gridDim.x;
    int th = blockDim.x * blockIdx.x + threadIdx.x;

    for (; th < height * width; th += block_size)
    {
        int lutx = lut[2 * th] >> fixed;
        int decx = lut[2 * th] & ((1 << fixed) - 1);
        int luty = lut[2 * th + 1] >> fixed;
        int decy = lut[2 * th + 1] & ((1 << fixed) - 1);

        int index = luty * src_width + lutx;

        int k0 = ((1 << fixed) - decy) * ((1 << fixed) - decx);  // a
        int k1 = decx * ((1 << fixed) - decy);  // b
        int k2 = ((1 << fixed) - decx) * decy;  // c
        int k3 = decx * decy;  // d

        int value = 0;
        if (index < height * width)
        {
            value += k0 * pd[index];
        }
        if (index + 1 < height * width)
        {
            value += k1 * pd[index + 1];
        }
        if (index + src_width < height * width)
        {
            value += k2 * pd[index + src_width];
        }
        if (index + src_width + 1 < height * width)
        {
            value += k3 * pd[index + src_width + 1];
        }
        value >>= 2 * fixed;
        p[th] = value;
    }
}

