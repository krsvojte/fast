cmake_minimum_required(VERSION 2.8)

project(Fast)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

#User provided hints for paths
list(APPEND CMAKE_PREFIX_PATH $ENV{GLFW_PATH})
list(APPEND CMAKE_PREFIX_PATH $ENV{GLEW_PATH})

add_subdirectory(fastlib)
add_subdirectory(fast)
add_subdirectory(fastviz)
