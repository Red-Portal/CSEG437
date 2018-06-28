
cmake_minimum_required(VERSION 3.8)
project(HW2)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# project source files
set(SOURCE_DIR ".")
set(SOURCE_FILES
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/thread_pool.cpp
    ${SOURCE_DIR}/cpu_implementation.cpp
    ${SOURCE_DIR}/shaders/LoadShaders.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Build options
if(NOT MSVC)
    set(BUILD_FLAGS
	"${CMAKE_CXX_FLAGS} -std=c++14 -g -O3 -Wall -Wextra -Wpedantic")
    find_package (Threads)
    target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
endif()

message("C++ flags = ${BUILD_FLAGS}")

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}")

find_package(OpenCL 1.2 REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(FreeImage REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenGL COMPONENTS GLX REQUIRED)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${FREEIMAGE_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${OpenGL_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME}
    ${FREEIMAGE_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCL_LIBRARY})

