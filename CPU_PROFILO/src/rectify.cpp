#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <rectify/rectify.h>

int const fixed = 2;

/*
 * Interpolazione bilineare in fixed point
*/
void rectify( std::vector<uint8_t> & p,
              const std::vector<uint8_t> & pd,
              const std::vector<int16_t> & lut,
              unsigned int const width,
              unsigned int const height,
              unsigned int const src_width)
{
    for (unsigned th{}; th < width * height; ++th)
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
        if (index < (int)(height * width))
        {
            value += k0 * pd[index];
        }
        if (index + 1 < (int)(height * width))
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
