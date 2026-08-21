cmake_minimum_required(VERSION 3.20)
project(DigitRecognitionProject)

set(CMAKE_CXX_STANDARD 17)

# Tell the compiler where to find all the header (.h) files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Process the source directory (this will build the core library and main engine)
add_subdirectory(src)

# Define the test executable
add_executable(test_recognizer tests/test_recognizer.cpp)

# Link the test executable to the core library (nn_core) built in the src directory
target_link_libraries(test_recognizer nn_core)