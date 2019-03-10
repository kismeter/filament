cmake_minimum_required(VERSION 3.1)
project(gltfio C ASM)

set(TARGET gltfio)
set(PUBLIC_HDR_DIR include)

# ==================================================================================================
# Sources and headers
# ==================================================================================================
set(PUBLIC_HDRS
        include/gltfio/Animator.h
        include/gltfio/AssetLoader.h
        include/gltfio/MaterialProvider.h
        include/gltfio/ResourceLoader.h
        include/gltfio/FilamentAsset.h
)

set(SRCS
        src/Animator.cpp
        src/AssetLoader.cpp
        src/FFilamentAsset.h
        src/FilamentAsset.cpp
        src/GltfEnums.h
        src/MaterialGenerator.cpp
        src/MaterialProvider.cpp
        src/ResourceLoader.cpp
        src/UbershaderLoader.cpp
        src/Wireframe.cpp
        src/Wireframe.h
        src/math.h
        src/upcast.h
)

# ==================================================================================================
# Build material
# ==================================================================================================

if (CMAKE_CROSSCOMPILING)
    include(${IMPORT_EXECUTABLES})
endif()

if (ANDROID OR WEBGL OR IOS)
    set(MATC_TARGET mobile)
else()
    set(MATC_TARGET desktop)
endif()

set(MATC_FLAGS -a all)
if (NOT CMAKE_BUILD_TYPE MATCHES Release)
    set(MATC_FLAGS -g ${MATC_FLAGS})
endif()

set(output_path "${CMAKE_CURRENT_BINARY_DIR}/ubershader.filamat")
set(input_path  "${CMAKE_CURRENT_SOURCE_DIR}/materials/ubershader.mat")

add_custom_command(
        OUTPUT ${output_path}
        COMMAND matc ${MATC_FLAGS} -p ${MATC_TARGET} -m material -o ${output_path} ${input_path}
        MAIN_DEPENDENCY ${input_path}
        DEPENDS matc
        COMMENT "Compiling material ${input_path} to ${output_path}"
)

set(MATERIAL_BINS ${output_path})

# ==================================================================================================
# Build resources library
# ==================================================================================================

set(RESOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR})

get_resgen_vars(${RESOURCE_DIR} gltfresources)

add_custom_command(
        OUTPUT ${RESGEN_OUTPUTS}
        COMMAND resgen ${RESGEN_FLAGS} ${MATERIAL_BINS}
        DEPENDS resgen ${MATERIAL_BINS}
)

if (DEFINED RESGEN_SOURCE_FLAGS)
    set_source_files_properties(${RESGEN_SOURCE} PROPERTIES COMPILE_FLAGS ${RESGEN_SOURCE_FLAGS})
endif()

set(DUMMY_SRC "${RESOURCE_DIR}/dummy.c")
add_custom_command(OUTPUT ${DUMMY_SRC} COMMAND echo "//" > ${DUMMY_SRC})

add_library(gltfio_resources ${DUMMY_SRC} ${RESGEN_SOURCE})

# ==================================================================================================
# Include and target definitions
# ==================================================================================================
include_directories(${PUBLIC_HDR_DIR} ${RESOURCE_DIR})

add_library(${TARGET} STATIC ${PUBLIC_HDRS} ${SRCS})

target_link_libraries(${TARGET} PUBLIC math utils filamat filament cgltf stb geometry gltfio_resources)

target_include_directories(${TARGET} PUBLIC ${PUBLIC_HDR_DIR})

# ==================================================================================================
# Compiler flags
# ==================================================================================================
target_compile_options(${TARGET} PRIVATE -Wno-deprecated-register)

if (MSVC OR CLANG_CL)
    target_compile_options(${TARGET} PRIVATE $<$<CONFIG:Release>:/fp:fast>)
else()
    target_compile_options(${TARGET} PRIVATE $<$<CONFIG:Release>:-ffast-math>)
endif()

# ==================================================================================================
# Installation
# ==================================================================================================
install(TARGETS ${TARGET} ARCHIVE DESTINATION lib/${DIST_DIR})
install(DIRECTORY ${PUBLIC_HDR_DIR}/gltfio DESTINATION include)
