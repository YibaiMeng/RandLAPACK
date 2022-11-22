set(RandLAPACK_cxx_srcs
    comps/determiter.cc
    comps/qb.cc
    comps/rf.cc
    comps/rs.cc
    comps/util.cc
    comps/orth.cc
    comps/rsvd.cc
)

set(RandLAPACK_libs
    RandLAPACK_RandBLAS
    RandLAPACK_lapackpp
    RandBLAS_blaspp
    RandBLAS_Random123
    RandBLAS_OpenMP
)

set_source_files_properties(${RandLAPACK_cxx_srcs} PROPERTIES LANGUAGE CUDA)
add_library(RandLAPACK ${RandLAPACK_cxx_srcs})
target_include_directories(
    RandLAPACK PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/RandLAPACK>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/RandLAPACK>
    ${Hamr_DIR}/include
)

target_link_libraries(RandLAPACK PUBLIC ${RandLAPACK_libs} hamr)
set_target_properties(RandLAPACK PROPERTIES LANGUAGE CUDA CUDA_ARCHITECTURES 61 CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 17)


install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
    DESTINATION include FILES_MATCHING PATTERN "*.hh"
)

install(
    TARGETS RandLAPACK
    EXPORT RandLAPACK
    INCLUDES DESTINATION include/RandLAPACK
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    EXPORT RandLAPACK
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
)
