
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

#include <rectify/rectify.h>
#include <census/census.h>
#include <preprocessor/pgm.h>

/**
 * Returns full path to current src file directory
 */
std::string getSrcFilePath()
{
    auto currentFile = std::string(__FILE__);

    auto pos = currentFile.rfind('\\');
    if (pos == std::string::npos)
    {
        pos = currentFile.rfind('/');
    }

    currentFile.erase(pos);
    return currentFile;
}

/**
 * Transform functor that turns a pair of pixels into one if they are different and to zero otherwise
 **/
struct TransformPixelFunctor
{
    __host__ __device__ std::size_t operator()(thrust::tuple<uint8_t, uint8_t> pixels)
    {
        if (thrust::get<0>(pixels) == thrust::get<1>(pixels))
        {
            return 0U;
        }
        else
        {
            return 1U;
        }
    }
};

int main(void)
{
    try
    {
    
        int rc = 0;
        unsigned int width = 0, height = 0;

        std::string repositoryFolder = getSrcFilePath() + std::string("/../"); // folder of repository
        std::string pictureFolder = repositoryFolder + std::string("res/"); // folder of pictures
        std::string buildFolder = repositoryFolder + std::string("build/"); // build of pictures
    
        thrust::device_vector<uint8_t> left_dist = pgm_load_gpu(width, height, pictureFolder + "left.pgm"); // load left picture
        thrust::device_vector<uint8_t> right_dist = pgm_load_gpu(width, height, pictureFolder + "right.pgm"); // load right picture
        thrust::device_vector<int16_t> left_lut = lut_load_fixed_gpu(width, height, pictureFolder + "left.blt"); // load left lut
        thrust::device_vector<int16_t> right_lut = lut_load_fixed_gpu(width, height, pictureFolder + "right.blt"); // load right lut

        thrust::device_vector<uint8_t> left(width * height); // buffer for left output picture
        thrust::device_vector<uint8_t> right(width * height); // buffer for right output picture

        cudaStream_t stream1, stream2; // two streams for performing calculations for left and right pictures in parallel

        // init two streams
        cudaStreamCreate( &stream1 );
        cudaStreamCreate( &stream2 );

        // perform rectification
        rectify(left,  left_dist,  left_lut,  width, height, width, stream1);
        rectify(right, right_dist, right_lut, width, height, width, stream2);

        // swap input and output picture buffers as image after rectification is an input to census
        left.swap(left_dist);
        right.swap(right_dist);

        // perform census
        census(left, left_dist, width, height, stream1);
        census(right, right_dist, width, height, stream2);

        // sync streams for starting census
        cudaStreamSynchronize( stream1 );
        cudaStreamSynchronize( stream2 );

        // desstroy streams as they are not needed anymore
        cudaStreamDestroy( stream1 );
        cudaStreamDestroy( stream2 );
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess)
        {
            std::cerr << "Warning: one or more CUDA errors occurred. ";
            std::cerr << "Try using cuda-gdb to debug. ";
            std::cerr << "Error message: \n\t" << cudaGetErrorString(cuda_error) << "\n";
            return 1;
        }

        // save output picture to ./build folder
        pgm_save_gpu(left, width, height, buildFolder + "left_processed.pgm");
        pgm_save_gpu(right, width, height, buildFolder + "right_processed.pgm");

        // creating zip iterators for proper work with thrust library
        const auto zipFirst = thrust::make_zip_iterator( thrust::make_tuple( std::begin(left), std::begin(right) ) );
        const auto zipLast = thrust::make_zip_iterator( thrust::make_tuple( std::end(left), std::end(right) ) );
    
        // count different pixels by firstly transforming a pair of pixels into 1 or 0 and then reducing it to a single value using plus operator
        const auto differentPixelsCount = thrust::transform_reduce(zipFirst, zipLast, TransformPixelFunctor{}, 0U, thrust::plus<std::size_t>{});

        // saving count of different pixels to the file
        save_different_pixels(differentPixelsCount, buildFolder + "different_pixels.txt");

        return rc;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

