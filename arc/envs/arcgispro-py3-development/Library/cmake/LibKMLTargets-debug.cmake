#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kmlbase" for configuration "Debug"
set_property(TARGET kmlbase APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(kmlbase PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C;CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/expat.lib;C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/z.lib;C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/minizip.lib;C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/uriparser.lib;C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/expat.lib"
  IMPORTED_LOCATION_DEBUG "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlbase.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS kmlbase )
list(APPEND _IMPORT_CHECK_FILES_FOR_kmlbase "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlbase.lib" )

# Import target "kmldom" for configuration "Debug"
set_property(TARGET kmldom APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(kmldom PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "kmlbase"
  IMPORTED_LOCATION_DEBUG "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmldom.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS kmldom )
list(APPEND _IMPORT_CHECK_FILES_FOR_kmldom "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmldom.lib" )

# Import target "kmlxsd" for configuration "Debug"
set_property(TARGET kmlxsd APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(kmlxsd PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "kmlbase"
  IMPORTED_LOCATION_DEBUG "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlxsd.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS kmlxsd )
list(APPEND _IMPORT_CHECK_FILES_FOR_kmlxsd "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlxsd.lib" )

# Import target "kmlengine" for configuration "Debug"
set_property(TARGET kmlengine APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(kmlengine PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "kmlbase;kmldom"
  IMPORTED_LOCATION_DEBUG "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlengine.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS kmlengine )
list(APPEND _IMPORT_CHECK_FILES_FOR_kmlengine "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlengine.lib" )

# Import target "kmlconvenience" for configuration "Debug"
set_property(TARGET kmlconvenience APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(kmlconvenience PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "kmlengine;kmldom;kmlbase"
  IMPORTED_LOCATION_DEBUG "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlconvenience.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS kmlconvenience )
list(APPEND _IMPORT_CHECK_FILES_FOR_kmlconvenience "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlconvenience.lib" )

# Import target "kmlregionator" for configuration "Debug"
set_property(TARGET kmlregionator APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(kmlregionator PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "kmlbase;kmldom;kmlengine;kmlconvenience"
  IMPORTED_LOCATION_DEBUG "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlregionator.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS kmlregionator )
list(APPEND _IMPORT_CHECK_FILES_FOR_kmlregionator "C:/Users/Lewis/Documents/GitHub/tenement-tools/arc/envs/arcgispro-py3-development/Library/lib/kmlregionator.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
