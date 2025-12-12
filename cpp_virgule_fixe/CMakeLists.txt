cmake_minimum_required(VERSION 3.15)
project(HardwareSimulator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_CORE CONFIGURE_DEPENDS src/*.cpp)
list(REMOVE_ITEM SRC_CORE ${CMAKE_CURRENT_SOURCE_DIR}/src/cnn_main.cpp)

add_library(cnn_floating STATIC ${SRC_CORE})
target_include_directories(cnn_floating PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(cnn src/cnn_main.cpp)
target_link_libraries(cnn PRIVATE cnn_floating)
target_include_directories(cnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()
file(GLOB TEST_FILES CONFIGURE_DEPENDS test/*.cpp)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${TEST_NAME} PRIVATE cnn_floating)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()