# cmake file for samples
cmake_minimum_required(VERSION 3.1)

project(SAMPLES LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD_REQUIRED 11)
set(CMAKE_CXX_STANDARD 11)
# disable non-standard compiler specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-DFINTEGER=int ")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3")

if(NOT (${BUILD_SAMPLE} MATCHES "ON"))
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
endif()

################################################################################
# neuware lib
################################################################################
if(NEUWARE_HOME)
  get_filename_component(NEUWARE_HOME ${NEUWARE_HOME} ABSOLUTE)
  message(STATUS "NEUWARE_HOME: ${NEUWARE_HOME}")
elseif(DEFINED ENV{NEUWARE_HOME})
  get_filename_component(NEUWARE_HOME $ENV{NEUWARE_HOME} ABSOLUTE)
  message(STATUS "ENV{NEUWARE_HOME}: ${NEUWARE_HOME}")
else()
  set(NEUWARE_HOME "/usr/local/neuware")
  message(STATUS "Default NEUWARE_HOME: ${NEUWARE_HOME}")
endif()

if((NOT EXISTS ${NEUWARE_HOME}) OR (NOT EXISTS ${NEUWARE_HOME}/include) OR (NOT EXISTS ${NEUWARE_HOME}/lib64))
  message(FATAL_ERROR "NEUWARE_HOME: ${NEUWARE_HOME} not exists!")
else()
  set(NEUWARE_INCLUDE_DIR ${NEUWARE_HOME}/include)
endif()

link_directories("${NEUWARE_HOME}/lib64" "${CMAKE_SOURCE_DIR}/../lib")


################################################################################
# compile
################################################################################
include_directories(${NEUWARE_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/../include)

add_executable(sample_flat sample_cnindex_flat.cpp)

target_include_directories(sample_flat PUBLIC
                           ${NEUWARE_INCLUDE_DIR} )
target_link_libraries(sample_flat PUBLIC cnindex cnnl cnnl_extra cnrt cndrv pthread)

add_executable(sample_ivfpq sample_cnindex_ivfpq.cpp)

target_include_directories(sample_ivfpq PUBLIC
                           ${NEUWARE_INCLUDE_DIR} )
target_link_libraries(sample_ivfpq PUBLIC cnindex cnnl cnnl_extra cnrt cndrv pthread)

add_executable(sample_pq sample_cnindex_pq.cpp)

target_include_directories(sample_pq PUBLIC
                           ${NEUWARE_INCLUDE_DIR} )
target_link_libraries(sample_pq PUBLIC cnindex cnnl cnnl_extra cnrt cndrv pthread)
