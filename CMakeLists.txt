cmake_minimum_required(VERSION 3.20)
project(DigitRecognitionProject)

set(CMAKE_CXX_STANDARD 17)

# אומר לקומפיילר איפה לחפש את קובצי ה-.h
include_directories(include)

# מפנה את מערכת הבנייה להיכנס לתיקיית src
add_subdirectory(src)