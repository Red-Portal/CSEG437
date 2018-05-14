
cmake_minimum_required(VERSION 3.5)
project(HW1)

# project source files
set(SOURCE_DIR ".")
set(SOURCE_FILES ${SOURCE_DIR}/main.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Build options
if(MSVC)
	add_definitions(-DHW_ROOT_PATH="${SOURCE_DIR}")
else()
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

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${OpenCL_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME}
    ${OpenCL_LIBRARY})
