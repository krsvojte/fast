##############################################
#### Global definitions
##############################################
cmake_minimum_required(VERSION 3.2)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)
project(fastviz LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")


#Set main directories
set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set (PROJECT_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set (IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)


get_directory_property(_has_parent PARENT_DIRECTORY)
if(_has_parent)
    set(TARGET_ROOT_DIR ${CMAKE_SOURCE_DIR})
else()
    set(TARGET_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()



##############################################
#### Configuration options
##############################################
option(GLEW_USE_STATIC_LIBS "Link static GLEW" ON)

 

##############################################
#### Dependencies
##############################################
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

#GLFW3
find_package(glfw3 REQUIRED)
if(GLFW3_FOUND)     
    message(STATUS "GLFW Found \n\tInclude: ${GLFW3_INCLUDE_DIR} \n\tLib:  ${GLFW3_LIBRARY}")  
else()
    message(WARNING "GLFW3 NOT FOUND,, specify enviromental var GLFW_PATH")
endif()



##############################################
#### Files
##############################################

set(_SRCS 
    ${PROJECT_SOURCE_DIR}/App.cpp
    ${PROJECT_SOURCE_DIR}/BatteryApp.cpp
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/Ui.cpp
    ${PROJECT_SOURCE_DIR}/render/Camera.cpp
    ${PROJECT_SOURCE_DIR}/render/Framebuffer.cpp
    ${PROJECT_SOURCE_DIR}/render/GLRenderer.cpp
    ${PROJECT_SOURCE_DIR}/render/Light.cpp
    ${PROJECT_SOURCE_DIR}/render/MeshObject.cpp
    ${PROJECT_SOURCE_DIR}/render/PrimitivesVBO.cpp
    ${PROJECT_SOURCE_DIR}/render/RenderList.cpp
    ${PROJECT_SOURCE_DIR}/render/Scene.cpp
    ${PROJECT_SOURCE_DIR}/render/Shader.cpp
    ${PROJECT_SOURCE_DIR}/render/ShaderResource.cpp
    ${PROJECT_SOURCE_DIR}/render/Shaders.cpp
    ${PROJECT_SOURCE_DIR}/render/Texture.cpp
    ${PROJECT_SOURCE_DIR}/render/VertexBuffer.cpp
    ${PROJECT_SOURCE_DIR}/render/VolumeRaycaster.cpp
    ${PROJECT_SOURCE_DIR}/utility/IOUtility.cpp
    ${PROJECT_SOURCE_DIR}/utility/Options.cpp  
    #external
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui.cpp
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui_draw.cpp
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui_file_explorer.cpp
    ${PROJECT_EXTERNAL_DIR}/imgui/imgui_impl_glfw_gl3.cpp
)

set(_HEADERS
	${PROJECT_INCLUDE_DIR}/App.h
    ${PROJECT_INCLUDE_DIR}/BatteryApp.h    
    ${PROJECT_INCLUDE_DIR}/Ui.h
    ${PROJECT_INCLUDE_DIR}/render/Camera.h
    ${PROJECT_INCLUDE_DIR}/render/Framebuffer.h
    ${PROJECT_INCLUDE_DIR}/render/GLRenderer.h
    ${PROJECT_INCLUDE_DIR}/render/Light.h
    ${PROJECT_INCLUDE_DIR}/render/MeshObject.h
    ${PROJECT_INCLUDE_DIR}/render/PrimitivesVBO.h
    ${PROJECT_INCLUDE_DIR}/render/RenderList.h
    ${PROJECT_INCLUDE_DIR}/render/Scene.h
    ${PROJECT_INCLUDE_DIR}/render/Shader.h
    ${PROJECT_INCLUDE_DIR}/render/ShaderResource.h
    ${PROJECT_INCLUDE_DIR}/render/Shaders.h
    ${PROJECT_INCLUDE_DIR}/render/Texture.h
    ${PROJECT_INCLUDE_DIR}/render/VertexBuffer.h
    ${PROJECT_INCLUDE_DIR}/render/VolumeRaycaster.h
    ${PROJECT_INCLUDE_DIR}/utility/IOUtility.h
    ${PROJECT_INCLUDE_DIR}/utility/Options.h
)

#internal include directories
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_EXTERNAL_DIR}")
include_directories("${IMGUI_DIR}")

#External

add_executable(fastviz 
    ${_SRCS} 
    ${_HEADERS} 
)


#add_definitions(
    #${fastlib_DEFINITIONS}
#)

add_dependencies(
    fastviz 
    fastlib
)

#Internal
#target_link_libraries(fastviz fastlib)

target_include_directories(fastviz  
  PRIVATE          
    fastlib
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    ${GLFW3_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}    
)

#External
target_link_libraries(fastviz
  PRIVATE
    fastlib
    ${GLFW3_LIBRARY}
    ${GLEW_LIBRARIES}
)

target_compile_definitions(fastviz  
  PRIVATE    
    GLM_FORCE_PURE   
    GLM_ENABLE_EXPERIMENTAL
    GLEW_STATIC
)


target_compile_options(fastviz
  PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/std:c++latest /openmp /MP>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-fopenmp -std=c++14 -lc -lm>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-fopenmp -std=c++14 -lc -lm>        
)


set_target_properties(fastviz
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${TARGET_ROOT_DIR}/bin
)

#target_link_libraries(fastviz ${GLFW3_LIBRARY})
#include_directories("${GLFW3_INCLUDE_DIR}")

#target_link_libraries(fastviz ${GLEW_LIBRARIES})
#include_directories("${GLEW_INCLUDE_DIRS}")

#include_directories("${EIGEN3_INCLUDE_DIR}")    #todo remove dependency


if(WIN32)
    set_target_properties(
        fastviz PROPERTIES 
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()

##############################################
#### Packaging
##############################################

install (TARGETS fastviz
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION bin)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/options.json" DESTINATION "fastviz")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders" DESTINATION "fastviz/src")

set(CPACK_PACKAGE_NAME "fastviz")
set(CPACK_PACKAGE_VERSION "1.0.0")

set(CPACK_MONOLITHIC_INSTALL 1)