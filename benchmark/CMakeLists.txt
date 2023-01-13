# TODO stop using gtest
message(STATUS "Checking for benchmarks ...")
enable_testing()
set(RandLAPACK_experiments_cxx_srcs
    experiments/comps/QB_cond_nums.cc 
    experiments/comps/Orth_speed.cc
    experiments/comps/rsvd_speed.cc
)
set_source_files_properties(${RandLAPACK_experiments_cxx_srcs} PROPERTIES LANGUAGE CUDA)
add_executable(RandLAPACK_benchmark ${RandLAPACK_experiments_cxx_srcs})
set_target_properties(RandLAPACK_benchmark PROPERTIES LANGUAGE CUDA CUDA_ARCHITECTURES 61)
target_link_libraries(RandLAPACK_benchmark PUBLIC OpenMP::OpenMP_CXX RandLAPACK GTest::gtest_main hamr)
include(GoogleTest)
gtest_discover_tests(RandLAPACK_benchmark)