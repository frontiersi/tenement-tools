#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "xdr-static" for configuration "Release"
set_property(TARGET xdr-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xdr-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libxdr.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS xdr-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_xdr-static "${_IMPORT_PREFIX}/lib/libxdr.lib" )

# Import target "xdr-shared" for configuration "Release"
set_property(TARGET xdr-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(xdr-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/xdr.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/xdr.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS xdr-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_xdr-shared "${_IMPORT_PREFIX}/lib/xdr.lib" "${_IMPORT_PREFIX}/bin/xdr.dll" )

# Import target "hdf-static" for configuration "Release"
set_property(TARGET hdf-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhdf.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf-static "${_IMPORT_PREFIX}/lib/libhdf.lib" )

# Import target "hdf-shared" for configuration "Release"
set_property(TARGET hdf-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hdf-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hdf.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hdf.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf-shared "${_IMPORT_PREFIX}/lib/hdf.lib" "${_IMPORT_PREFIX}/bin/hdf.dll" )

# Import target "mfhdf-static" for configuration "Release"
set_property(TARGET mfhdf-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mfhdf-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmfhdf.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS mfhdf-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_mfhdf-static "${_IMPORT_PREFIX}/lib/libmfhdf.lib" )

# Import target "mfhdf-shared" for configuration "Release"
set_property(TARGET mfhdf-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mfhdf-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/mfhdf.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/mfhdf.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS mfhdf-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_mfhdf-shared "${_IMPORT_PREFIX}/lib/mfhdf.lib" "${_IMPORT_PREFIX}/bin/mfhdf.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
