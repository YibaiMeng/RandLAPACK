cmake_minimum_required(VERSION 3.2)
SET(CUDA_SEPARABLE_COMPILATION ON)
project(RandLAPACK LANGUAGES CUDA CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# configure the build
enable_testing()

include(build_options)
include(version)

# find dependencies
include(RandLAPACK_RandBLAS)
include(lapackpp)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# compile sources
enable_language(CUDA)
add_compile_definitions(HAMR_ENABLE_CUDA)
include(${Hamr_DIR}/lib/hamr.cmake)
#include(/home/eecs/mengyibai/proj/BallisticLA/HAMR-install/lib/hamr-release.cmake)
# TODO: find a way to enable this automatically
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)

# export the configuration
include(config)
