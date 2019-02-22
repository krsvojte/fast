##############################################
#### Global definitions
##############################################
cmake_minimum_required(VERSION 3.2)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)
project(fastlib LANGUAGES CXX CUDA)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

##############################################
#### Configuration options
##############################################

option(GLEW_USE_STATIC_LIBS "Link static GLEW" ON)
set(CUDA_ARCH "-arch=sm_30" CACHE STRING "NVCC flags")

get_directory_property(_has_parent PARENT_DIRECTORY)
if(_has_parent)
    set(TARGET_ROOT_DIR ${CMAKE_SOURCE_DIR})
else()
    set(TARGET_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()


##############################################
#### Dependencies
##############################################
list(APPEND CMAKE_PREFIX_PATH $ENV{EIGEN_PATH})
find_package (Eigen3 3.1 REQUIRED)
if(EIGEN3_NOTFOUND)            
    message(WARNING "EIGEN3 NOT FOUND, specify enviromental var EIGEN_PATH")
endif()

#OpenGL
find_package(OpenGL REQUIRED)
if(OPENGL_FOUND)		
	link_libraries(${OPENGL_LIBRARIES})	
	if(OPENGL_INCLUDE_DIR)
		include_directories("${OPENGL_INCLUDE_DIR}")
	endif()
	message(STATUS "OpenGL Found \n\tInclude: ${OPENGL_INCLUDE_DIR} \n\tLib:  ${OPENGL_gl_LIBRARY}")
else()
	message(WARNING "OpenGL NOT FOUND")
endif()

#GLEW
if (WIN32)
set (GLEW_ROOT_DIR $ENV{GLEW_PATH})
endif()


find_package(GLEW REQUIRED)
if(GLEW_FOUND)    	
	message(STATUS "GLEW Found \n\tInclude: ${GLEW_INCLUDE_DIRS} \n\tLib:  ${GLEW_LIBRARIES}")  
else()
	message(WARNING "GLEW NOT FOUND, specify enviromental var GLEW_PATH")
endif()


#Cudart
if(DEFINED ENV{CUDA_PATH})
	set(CUDA_PATH $ENV{CUDA_PATH})
	string(REPLACE "\\" "/" CUDA_PATH ${CUDA_PATH})
	set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDA_PATH}/include)
	message(STATUS "CUDA Found \n\tInclude: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")  
else()
	message(WARNING "CUDA_PATH NOT FOUND, specify enviromental var CUDA_PATH")
endif()
#Cuda specific properties
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(OpenMP REQUIRED)


##############################################
#### Files
##############################################

set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(
     GLOB_RECURSE source_list RELATIVE
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${PROJECT_SOURCE_DIR}/*.cpp"
     "${PROJECT_SOURCE_DIR}/*.h"
     "${PROJECT_SOURCE_DIR}/*.cu"
     "${PROJECT_SOURCE_DIR}/*.cuh"
     "${PROJECT_SOURCE_DIR}/*.hpp"            
    )

#Includes
file(
     GLOB_RECURSE includes RELATIVE
     "${CMAKE_CURRENT_SOURCE_DIR}"
     #Includes  
     "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"  
     "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
    )

#External compiled files
file(
	GLOB external_src RELATIVE
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/external/TinyTIFF/*.cpp"  
    "${CMAKE_CURRENT_SOURCE_DIR}/external/TinyTIFF/*.h"    
	"${CMAKE_CURRENT_SOURCE_DIR}/external/quickhull/*.cpp"  
    "${CMAKE_CURRENT_SOURCE_DIR}/external/quickhull/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/distfun/*.hpp"
)

list(APPEND source_list ${external_src})
list(APPEND source_list ${includes})


if (WIN32)
    foreach(source IN LISTS source_list)
        get_filename_component(source_path "${source}" PATH)
        string(REPLACE "/" "\\" source_path_msvc "${source_path}")
        source_group("${source_path_msvc}" FILES "${source}")
    endforeach()
endif()


##############################################
#### Target definitions
##############################################

add_library(fastlib SHARED 
    ${source_list}    
)

add_library(fast::fastlib ALIAS fastlib)

set_target_properties(fastlib 
	PROPERTIES 
		CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(fastlib
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/glm>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${GLEW_INCLUDE_DIRS}
  PRIVATE    
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/fastlib>    
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>    
    ${EIGEN3_INCLUDE_DIR}    
)

target_link_libraries(fastlib
  PUBLIC
    ${GLEW_LIBRARIES}
    OpenMP::OpenMP_CXX
)


target_compile_definitions(fastlib
  PUBLIC    
    FASTLIB_SHARED
    GLM_ENABLE_EXPERIMENTAL
    GLEW_STATIC
  PRIVATE
    FASTLIB_EXPORT 
    GLM_FORCE_PURE           
)

target_compile_options(fastlib
  PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/std:c++latest /MP>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-std=c++14 -lc -lm -Wno-deprecated>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-std=c++14 -lc -lm>    
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_ARCH}>
)

set_target_properties(fastlib
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/bin
)




##############################################
#### Packaging
##############################################

install (TARGETS fastlib
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION bin)

install(DIRECTORY include DESTINATION "./")


set(CPACK_PACKAGE_NAME "fastlib")
set(CPACK_PACKAGE_VERSION "1.0.0")

set(CPACK_MONOLITHIC_INSTALL 1)

include(CPack)



