cmake_minimum_required(VERSION 3.16)

project(hprqp_c VERSION 0.1.0 LANGUAGES C CXX)

option(HPRQP_ENABLE_CUDA "Build the CUDA backend scaffold" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release flags for C" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release flags for C++" FORCE)

set(HPRQP_SHARED_SOURCES
    src/shared/binding.c
    src/shared/build.c
    src/shared/mps_qp.cpp
    src/shared/mps_reader.cpp
    src/shared/parallel.c
    src/shared/solver.c
    src/shared/solver_backend.c
    src/shared/sparse.c
    src/shared/vector.c
)

set(HPRQP_CPU_SOURCES
    src/cpu/control.c
    src/cpu/eigs.c
    src/cpu/iteration.c
    src/cpu/residuals.c
    src/cpu/scale.c
    src/cpu/workspace.c
)

set(HPRQP_GPU_SOURCES
    src/gpu/backend_stub.c
    src/gpu/solver_stub.c
)

if(HPRQP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "HPRQP_ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release flags for CUDA" FORCE)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")

    string(REGEX MATCH "^[0-9]+\\.[0-9]+" HPRQP_NVCC_VERSION_MM "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" HPRQP_TOOLKIT_VERSION_MM "${CUDAToolkit_VERSION}")
    if(HPRQP_NVCC_VERSION_MM AND HPRQP_TOOLKIT_VERSION_MM AND NOT HPRQP_NVCC_VERSION_MM STREQUAL HPRQP_TOOLKIT_VERSION_MM)
        message(FATAL_ERROR
            "Detected a mixed CUDA toolchain: nvcc reports ${CMAKE_CUDA_COMPILER_VERSION} "
            "but CUDAToolkit reports ${CUDAToolkit_VERSION}. This can cause generated stub "
            "compile failures such as '__cudaLaunch requires 2 arguments'. Ensure PATH, "
            "CUDAToolkit_ROOT, and CMAKE_CUDA_COMPILER all point to the same CUDA installation.")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(HPRQP_DETECTED_CUDA_ARCHITECTURES "")
        find_program(HPRQP_NVIDIA_SMI_EXECUTABLE nvidia-smi)
        if(HPRQP_NVIDIA_SMI_EXECUTABLE)
            execute_process(
                COMMAND ${HPRQP_NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader,nounits
                OUTPUT_VARIABLE HPRQP_NVIDIA_SMI_ARCHS_RAW
                RESULT_VARIABLE HPRQP_NVIDIA_SMI_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(HPRQP_NVIDIA_SMI_RESULT EQUAL 0 AND NOT HPRQP_NVIDIA_SMI_ARCHS_RAW STREQUAL "")
                string(REPLACE "\n" ";" HPRQP_NVIDIA_SMI_ARCH_LINES "${HPRQP_NVIDIA_SMI_ARCHS_RAW}")
                foreach(HPRQP_ARCH_LINE IN LISTS HPRQP_NVIDIA_SMI_ARCH_LINES)
                    string(STRIP "${HPRQP_ARCH_LINE}" HPRQP_ARCH_LINE)
                    if(NOT HPRQP_ARCH_LINE STREQUAL "")
                        string(REPLACE "." "" HPRQP_ARCH_VALUE "${HPRQP_ARCH_LINE}")
                        list(APPEND HPRQP_DETECTED_CUDA_ARCHITECTURES "${HPRQP_ARCH_VALUE}")
                    endif()
                endforeach()
            endif()
        endif()

        if(HPRQP_DETECTED_CUDA_ARCHITECTURES)
            list(REMOVE_DUPLICATES HPRQP_DETECTED_CUDA_ARCHITECTURES)
            list(SORT HPRQP_DETECTED_CUDA_ARCHITECTURES COMPARE NATURAL)
            set(CMAKE_CUDA_ARCHITECTURES "${HPRQP_DETECTED_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures for hprqp" FORCE)
            message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90" CACHE STRING "CUDA architectures for hprqp" FORCE)
            message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    set(HPRQP_GPU_SOURCES
        src/gpu/context.cu
        src/gpu/workspace.cu
        src/gpu/scale.cu
        src/gpu/spmv.cu
        src/gpu/eigs.cu
        src/gpu/control.cu
        src/gpu/residuals.cu
        src/gpu/qcqp.cu
        src/gpu/iteration.cu
        src/gpu/solver.cu
    )
endif()

find_library(MATH_LIBRARY m)

add_library(hprqp
    ${HPRQP_SHARED_SOURCES}
    ${HPRQP_CPU_SOURCES}
    ${HPRQP_GPU_SOURCES}
)

add_library(hprqp::hprqp ALIAS hprqp)

target_include_directories(hprqp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shared
)

set_target_properties(hprqp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME hprqp
)

target_compile_options(hprqp
    PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
)

if(MATH_LIBRARY)
    target_link_libraries(hprqp PUBLIC ${MATH_LIBRARY})
endif()

if(HPRQP_ENABLE_CUDA)
    set_property(TARGET hprqp PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(hprqp PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
    target_compile_definitions(hprqp PUBLIC HPRQP_HAVE_CUDA=1)
else()
    target_compile_definitions(hprqp PUBLIC HPRQP_HAVE_CUDA=0)
endif()

add_library(hprqp_test_support
    tests/test_support.c
)

target_include_directories(hprqp_test_support
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

target_link_libraries(hprqp_test_support PUBLIC hprqp)

add_executable(test_sparse_helpers
    tests/test_sparse_helpers.c
)

target_link_libraries(test_sparse_helpers PRIVATE hprqp_test_support)

add_executable(test_mps_reader
    tests/test_mps_reader.c
)

target_link_libraries(test_mps_reader PRIVATE hprqp_test_support)
target_compile_definitions(test_mps_reader PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_builder_preprocess
    tests/test_builder_preprocess.c
)

target_link_libraries(test_builder_preprocess PRIVATE hprqp_test_support)

add_executable(test_binding_api
    tests/test_binding_api.c
)

target_link_libraries(test_binding_api PRIVATE hprqp)
target_compile_definitions(test_binding_api PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_workspace
    tests/cpu/test_workspace.c
)

target_link_libraries(test_workspace PRIVATE hprqp_test_support)

add_executable(test_scale
    tests/cpu/test_scale.c
)

target_link_libraries(test_scale PRIVATE hprqp_test_support)

add_executable(test_residuals
    tests/cpu/test_residuals.c
)

target_link_libraries(test_residuals PRIVATE hprqp_test_support)

add_executable(test_solver
    tests/cpu/test_solver.c
)

target_link_libraries(test_solver PRIVATE hprqp_test_support)

add_executable(test_iteration
    tests/cpu/test_iteration.c
)

target_link_libraries(test_iteration PRIVATE hprqp_test_support)

add_executable(test_parallel
    tests/cpu/test_parallel.c
)

target_link_libraries(test_parallel PRIVATE hprqp_test_support)

add_executable(test_eigs
    tests/cpu/test_eigs.c
)

target_link_libraries(test_eigs PRIVATE hprqp_test_support)

add_executable(test_control
    tests/cpu/test_control.c
)

target_link_libraries(test_control PRIVATE hprqp_test_support)

add_executable(test_gpu_dispatch
    tests/gpu/test_gpu_dispatch.c
)

target_link_libraries(test_gpu_dispatch PRIVATE hprqp_test_support)

if(HPRQP_ENABLE_CUDA)
    add_executable(test_binding_api_gpu
        tests/gpu/test_binding_api_gpu.c
    )

    target_link_libraries(test_binding_api_gpu PRIVATE hprqp)

    add_executable(test_gpu_workspace
        tests/gpu/test_gpu_workspace.c
    )

    target_link_libraries(test_gpu_workspace PRIVATE hprqp_test_support)

    add_executable(test_gpu_scale
        tests/gpu/test_gpu_scale.c
    )

    target_link_libraries(test_gpu_scale PRIVATE hprqp_test_support)

    add_executable(test_gpu_residuals
        tests/gpu/test_gpu_residuals.c
    )

    target_link_libraries(test_gpu_residuals PRIVATE hprqp_test_support)

    add_executable(test_gpu_eigs
        tests/gpu/test_gpu_eigs.c
    )

    target_link_libraries(test_gpu_eigs PRIVATE hprqp_test_support)

    add_executable(test_gpu_control
        tests/gpu/test_gpu_control.c
    )

    target_link_libraries(test_gpu_control PRIVATE hprqp_test_support)

    add_executable(test_gpu_qcqp
        tests/gpu/test_gpu_qcqp.c
    )

    target_link_libraries(test_gpu_qcqp PRIVATE hprqp_test_support)
    target_compile_definitions(test_gpu_qcqp PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(test_gpu_iteration
        tests/gpu/test_gpu_iteration.cpp
    )

    target_link_libraries(test_gpu_iteration PRIVATE hprqp_test_support)

    add_executable(test_gpu_solver
        tests/gpu/test_gpu_solver.c
    )

    target_link_libraries(test_gpu_solver PRIVATE hprqp_test_support)

    add_executable(compare_gpu_unchecked_path
        examples/gpu/compare_gpu_unchecked_path.c
    )

    target_link_libraries(compare_gpu_unchecked_path PRIVATE hprqp)

    add_executable(trace_restart_compare
        examples/gpu/trace_restart_compare.c
    )

    target_link_libraries(trace_restart_compare PRIVATE hprqp)
endif()

add_executable(solve_qp
    examples/solve_qp.c
)

target_link_libraries(solve_qp PRIVATE hprqp)
target_compile_definitions(solve_qp PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

enable_testing()
add_test(NAME test_sparse_helpers COMMAND test_sparse_helpers)
add_test(NAME test_mps_reader COMMAND test_mps_reader)
add_test(NAME test_builder_preprocess COMMAND test_builder_preprocess)
add_test(NAME test_binding_api COMMAND test_binding_api)
add_test(NAME test_workspace COMMAND test_workspace)
add_test(NAME test_scale COMMAND test_scale)
add_test(NAME test_residuals COMMAND test_residuals)
add_test(NAME test_solver COMMAND test_solver)
add_test(NAME test_iteration COMMAND test_iteration)
add_test(NAME test_parallel COMMAND test_parallel)
add_test(NAME test_eigs COMMAND test_eigs)
add_test(NAME test_control COMMAND test_control)
add_test(NAME test_gpu_dispatch COMMAND test_gpu_dispatch)
if(HPRQP_ENABLE_CUDA)
    add_test(NAME test_binding_api_gpu COMMAND test_binding_api_gpu)
    add_test(NAME test_gpu_workspace COMMAND test_gpu_workspace)
    add_test(NAME test_gpu_scale COMMAND test_gpu_scale)
    add_test(NAME test_gpu_residuals COMMAND test_gpu_residuals)
    add_test(NAME test_gpu_eigs COMMAND test_gpu_eigs)
    add_test(NAME test_gpu_control COMMAND test_gpu_control)
    add_test(NAME test_gpu_qcqp COMMAND test_gpu_qcqp)
    add_test(NAME test_gpu_iteration COMMAND test_gpu_iteration)
    add_test(NAME test_gpu_solver COMMAND test_gpu_solver)
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hprqpConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hprqp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS hprqp
    EXPORT hprqpTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hprqp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hprqp_binding.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shared
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT hprqpTargets
    FILE hprqpTargets.cmake
    NAMESPACE hprqp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hprqp
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hprqp
)
