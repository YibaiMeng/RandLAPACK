# TODO stop using gtest
message(STATUS "Checking for benchmarks ...")
find_package(GTest)
set(tmp FALSE)
if (GTest_FOUND)
    set(tmp TRUE)
    set(RandLAPACK_experiments_cxx_srcs
        experiments/comps/QB_cond_nums.cc 
        experiments/comps/Orth_speed.cc
        experiments/comps/rsvd_speed.cc
    )
    set_source_files_properties(${RandLAPACK_experiments_cxx_srcs} PROPERTIES LANGUAGE CUDA)
    add_executable(RandLAPACK_benchmark ${RandLAPACK_experiments_cxx_srcs})
    set_target_properties(RandLAPACK_benchmark PROPERTIES LANGUAGE CUDA CUDA_ARCHITECTURES 61)
    target_link_libraries(RandLAPACK_benchmark PUBLIC OpenMP::OpenMP_CXX RandLAPACK GTest::GTest GTest::Main hamr)
    gtest_discover_tests(RandLAPACK_benchmark)

endif()
message(STATUS "Checking for benchmarks ... ${tmp}")