
#include <algorithm>
#include <chrono>
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

int main(void)
{
    try
    {
    
        int rc = 0;
        unsigned int width = 0, height = 0;

        std::string repositoryFolder = getSrcFilePath() + std::string("/../"); // folder of repository
        std::string pictureFolder = repositoryFolder + std::string("res/"); // folder of pictures
        std::string buildFolder = repositoryFolder + std::string("build/"); // build of pictures
    
        thrust::device_vector<uint8_t> left_dist_init = pgm_load_gpu(width, height, pictureFolder + "left.pgm"); // load left picture
        thrust::device_vector<uint8_t> right_dist_init = pgm_load_gpu(width, height, pictureFolder + "right.pgm"); // load right picture
    
        thrust::device_vector<uint8_t> left_dist = left_dist_init;
        thrust::device_vector<uint8_t> right_dist = right_dist_init;
        thrust::device_vector<int16_t> left_lut = lut_load_fixed_gpu(width, height, pictureFolder + "left.blt"); // load left lut
        thrust::device_vector<int16_t> right_lut = lut_load_fixed_gpu(width, height, pictureFolder + "right.blt"); // load right lut

        thrust::device_vector<uint8_t> left(width * height); // buffer for left output picture
        thrust::device_vector<uint8_t> right(width * height); // buffer for right output picture

        cudaStream_t stream1, stream2; // two streams for performing calculations for left and right pictures in parallel

        // init two streams
        cudaStreamCreate( &stream1 );
        cudaStreamCreate( &stream2 );

        std::size_t minMicroSeconds = std::numeric_limits<std::size_t>::max();
        std::size_t iterationCount = 1000U;
        for (std::size_t i{}; i < iterationCount; ++i)
        {
            left_dist = left_dist_init;
            right_dist = right_dist_init;

            const auto start = std::chrono::high_resolution_clock::now();
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
            cudaError_t cuda_error = cudaGetLastError();
            if (cuda_error != cudaSuccess)
            {
                printf("Warning: one or more CUDA errors occurred. ");
                printf("Try using cuda-gdb to debug. ");
                printf("Error message: \n\t%s\n", cudaGetErrorString(cuda_error));
                return 1;
            }
            const auto end = std::chrono::high_resolution_clock::now();

            minMicroSeconds = std::min<std::size_t>(minMicroSeconds, std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        }

        std::cout << "algorithm time is: " << minMicroSeconds << " microseconds.\n";

        // desstroy streams as they are not needed anymore
        cudaStreamDestroy( stream1 );
        cudaStreamDestroy( stream2 );

        // save output picture to ./build folder
        pgm_save_gpu(left, width, height, buildFolder + "left_rect.pgm");
        pgm_save_gpu(right, width, height, buildFolder + "right_rect.pgm");

        return rc;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

