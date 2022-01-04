
cmake_minimum_required(VERSION 3.10)
project(main LANGUAGES CXX CUDA)

include_directories(./include)

add_executable(main ./src/main.cu
                    ./src/pgm.cu
                    ./src/rectify.cu
                    ./src/census.cu)
