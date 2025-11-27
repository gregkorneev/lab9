cmake_minimum_required(VERSION 3.10)

# Название проекта
project(lab14_hyperopt LANGUAGES CXX)

# Стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Режим сборки по умолчанию
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Подключаем папку с заголовками
include_directories(${CMAKE_SOURCE_DIR}/include)

# Исполняемый файл и исходники
add_executable(lab14_hyperopt
    src/main.cpp
    src/model.cpp
    src/algorithms.cpp
)
