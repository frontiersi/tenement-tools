#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "geotiff_library" for configuration "Release"
set_property(TARGET geotiff_library APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(geotiff_library PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/geotiff_i.lib"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/tiff.lib;PROJ::proj;C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/z.lib;C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/libjpeg.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/geotiff.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS geotiff_library )
list(APPEND _IMPORT_CHECK_FILES_FOR_geotiff_library "${_IMPORT_PREFIX}/lib/geotiff_i.lib" "${_IMPORT_PREFIX}/bin/geotiff.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
