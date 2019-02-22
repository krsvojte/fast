##############################################
#### Global definitions
##############################################
cmake_minimum_required(VERSION 3.2)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

project(fast LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")




get_directory_property(_has_parent PARENT_DIRECTORY)
if(_has_parent)
    set(TARGET_ROOT_DIR ${CMAKE_SOURCE_DIR})
else()
    set(TARGET_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()




##############################################
#### Files
##############################################

#Set main directories
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE SRC_CPP
    "${PROJECT_SOURCE_DIR}/main.cpp"    
)

file(
     GLOB_RECURSE SRC_H 
     "${PROJECT_SOURCE_DIR}/*.h"       
    )



source_group("Source"       FILES ${SRC_CPP})
source_group("Headers"      FILES ${SRC_H})

#internal include directories

add_executable(fast ${SRC_CPP} ${SRC_H})

add_dependencies(fast fastlib)

target_link_libraries(fast
	PRIVATE
		fastlib		
)

target_compile_options(fast
  PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/std:c++latest /MP>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-std=c++14 -lc -lm -Wno-deprecated -Wl,-R../lib/>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-std=c++14 -lc -lm -Wl,-R../lib/>        
)

set_target_properties(fast
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/bin
)

target_include_directories(fast  
  PRIVATE              
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>    
)
#External
#target_link_libraries(fast ${GLEW_LIBRARIES})
#include_directories("${GLEW_INCLUDE_DIRS}")


if(WIN32)
set_target_properties(fast PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()


install (TARGETS fast
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION bin)


set(CPACK_PACKAGE_NAME "fast")
set(CPACK_PACKAGE_VERSION "1.0.0")

set(CPACK_MONOLITHIC_INSTALL 1)

include(CPack)
