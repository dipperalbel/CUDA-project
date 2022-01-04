
cmake_minimum_required(VERSION 3.10)
project(main LANGUAGES CXX CUDA)

include_directories(./include)

add_executable(main ./src/main.cpp
                    ./src/pgm.cpp
                    ./src/rectify.cpp
                    ./src/census.cpp)
