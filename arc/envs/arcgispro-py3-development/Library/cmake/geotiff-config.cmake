# Configure GeoTIFF
#
# Set
#  GeoTIFF_FOUND = 1
#  GeoTIFF_INCLUDE_DIRS = /usr/local/include
#  GeoTIFF_SHARED_LIBRARIES = geotiff_library
#  GeoTIFF_STATIC_LIBRARIES = geotiff_archive
#  GeoTIFF_LIBRARY_DIRS = /usr/local/lib
#  GeoTIFF_BINARY_DIRS = /usr/local/bin
#  GeoTIFF_VERSION = 1.4.1 (for example)
#  Depending on GeoTIFF_USE_STATIC_LIBS
#    GeoTIFF_LIBRARIES = ${GeoTIFF_SHARED_LIBRARIES}, if OFF
#    GeoTIFF_LIBRARIES = ${GeoTIFF_STATIC_LIBRARIES}, if ON

# For compatibility with FindGeoTIFF.cmake, also set
# GEOTIFF_FOUND
# GEOTIFF_INCLUDE_DIR
# GEOTIFF_LIBRARY
# GEOTIFF_LIBRARIES

message (STATUS "Reading ${CMAKE_CURRENT_LIST_FILE}")
# GeoTIFF_VERSION is set by version file
message (STATUS
  "GeoTIFF configuration, version ${GeoTIFF_VERSION}")

# Tell the user project where to find our headers and libraries
get_filename_component (_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
get_filename_component (_ROOT "${_DIR}/.." ABSOLUTE)
set (GeoTIFF_INCLUDE_DIRS "${_ROOT}/include")
set (GeoTIFF_LIBRARY_DIRS "${_ROOT}/lib")
set (GeoTIFF_BINARY_DIRS "${_ROOT}/bin")

message (STATUS "  include directory: \${GeoTIFF_INCLUDE_DIRS}")

if(BUILD_SHARED_LIBS)
set (GeoTIFF_SHARED_LIBRARIES geotiff_library)
else()
set (GeoTIFF_STATIC_LIBRARIES geotiff_library)
endif()
# Read in the exported definition of the library
include ("${_DIR}/geotiff-depends.cmake")

unset (_ROOT)
unset (_DIR)

if (GeoTIFF_USE_STATIC_LIBS)
  set (GeoTIFF_LIBRARIES ${GeoTIFF_STATIC_LIBRARIES})
  message (STATUS "  \${GeoTIFF_LIBRARIES} set to static library")
else ()
  set (GeoTIFF_LIBRARIES ${GeoTIFF_SHARED_LIBRARIES})
  message (STATUS "  \${GeoTIFF_LIBRARIES} set to shared library")
endif ()

# For compatibility with FindGeoTIFF.cmake
set (GEOTIFF_FOUND 1)
set (GEOTIFF_LIBRARIES ${GeoTIFF_LIBRARIES})
set (GEOTIFF_INCLUDE_DIR ${GeoTIFF_INCLUDE_DIRS})
set (GEOTIFF_LIBRARY ${GeoTIFF_LIBRARIES})
