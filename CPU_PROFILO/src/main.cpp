
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
    
        const std::vector<uint8_t> left_dist_init = pgm_load(width, height, pictureFolder + "left.pgm"); // load left picture
        const std::vector<uint8_t> right_dist_init = pgm_load(width, height, pictureFolder + "right.pgm"); // load right picture

        std::vector<uint8_t> left_dist = left_dist_init;
        std::vector<uint8_t> right_dist = right_dist_init;
        std::vector<int16_t> left_lut = lut_load_fixed(width, height, pictureFolder + "left.blt"); // load left lut
        std::vector<int16_t> right_lut = lut_load_fixed(width, height, pictureFolder + "right.blt"); // load right lut

        std::vector<uint8_t> left(width * height); // buffer for left output picture
        std::vector<uint8_t> right(width * height); // buffer for right output picture

        std::size_t minMicroSeconds = std::numeric_limits<std::size_t>::max();
        std::size_t iterationCount = 100U;
        for (std::size_t i{}; i < iterationCount; ++i)
        {
            left_dist = left_dist_init;
            right_dist = right_dist_init;

            const auto start = std::chrono::high_resolution_clock::now();
            // perform rectification
            rectify(left,  left_dist,  left_lut,  width, height, width);
            rectify(right, right_dist, right_lut, width, height, width);

            // swap input and output picture buffers as image after rectification is an input to census
            left.swap(left_dist);
            right.swap(right_dist);

            // perform census
            census(left, left_dist, width, height);
            census(right, right_dist, width, height);
            const auto end = std::chrono::high_resolution_clock::now();

            minMicroSeconds = std::min<std::size_t>(minMicroSeconds, std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        }

        std::cout << "algorithm time is: " << minMicroSeconds << " microseconds.\n";

        // save output picture to ./build folder
        pgm_save(left, width, height, buildFolder + "left_rect.pgm");
        pgm_save(right, width, height, buildFolder + "right_rect.pgm");

        return rc;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

