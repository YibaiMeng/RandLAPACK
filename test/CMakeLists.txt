message(STATUS "Checking for regression tests ...")
enable_testing()
set(RandLAPACK_testx_cxx_srcs
    src/comps/test_determiter.cc 
    src/comps/test_util.cc 
    src/comps/test_orth.cc 
    src/comps/test_qb.cc
    src/comps/test_rsvd.cc
)
set_source_files_properties(${RandLAPACK_testx_cxx_srcs} PROPERTIES LANGUAGE CUDA)
add_executable(RandLAPACK_tests ${RandLAPACK_testx_cxx_srcs})
set_target_properties(RandLAPACK_tests PROPERTIES LANGUAGE CUDA CUDA_ARCHITECTURES 61 CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(RandLAPACK_tests PUBLIC OpenMP::OpenMP_CXX RandLAPACK GTest::gtest_main hamr)
include(GoogleTest)
gtest_discover_tests(RandLAPACK_tests)