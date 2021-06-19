#-----------------------------------------------------------------------------
# HDF4 Config file for compiling against hdf4 build/install directory
#-----------------------------------------------------------------------------

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was hdf4-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

string(TOUPPER hdf4 HDF4_PACKAGE_NAME)

set (${HDF4_PACKAGE_NAME}_VALID_COMPONENTS
    static
    shared
    C
    Fortran
    Java
)

#-----------------------------------------------------------------------------
# User Options
#-----------------------------------------------------------------------------
set (${HDF4_PACKAGE_NAME}_BUILD_FORTRAN   NO)
set (${HDF4_PACKAGE_NAME}_BUILD_JAVA      OFF)
set (${HDF4_PACKAGE_NAME}_BUILD_XDR_LIB   ON)
set (${HDF4_PACKAGE_NAME}_BUILD_TOOLS     OFF)
set (${HDF4_PACKAGE_NAME}_BUILD_UTILS     OFF)
set (${HDF4_PACKAGE_NAME}_ENABLE_JPEG_LIB_SUPPORT ON)
set (${HDF4_PACKAGE_NAME}_ENABLE_Z_LIB_SUPPORT ON)
set (${HDF4_PACKAGE_NAME}_ENABLE_SZIP_SUPPORT  OFF)
set (${HDF4_PACKAGE_NAME}_ENABLE_SZIP_ENCODING )
set (${HDF4_PACKAGE_NAME}_BUILD_SHARED_LIBS    YES)
set (${HDF4_PACKAGE_NAME}_BUILD_STATIC_LIBS    YES)
set (${HDF4_PACKAGE_NAME}_PACKAGE_EXTLIBS      OFF)
set (${HDF4_PACKAGE_NAME}_EXPORT_LIBRARIES xdr-static;xdr-shared;hdf-static;hdf-shared;mfhdf-static;mfhdf-shared)
set (${HDF4_PACKAGE_NAME}_TOOLSET "")

#-----------------------------------------------------------------------------
# Dependencies
#-----------------------------------------------------------------------------
if (${HDF4_PACKAGE_NAME}_BUILD_JAVA)
  set (${HDF4_PACKAGE_NAME}_JAVA_INCLUDE_DIRS
      ${PACKAGE_PREFIX_DIR}/lib/jarhdf-4.2.15.jar
      ${PACKAGE_PREFIX_DIR}/lib/slf4j-api-1.7.25.jar
      ${PACKAGE_PREFIX_DIR}/lib/slf4j-nop-1.7.25.jar
  )
  set (${HDF4_PACKAGE_NAME}_JAVA_LIBRARY "${PACKAGE_PREFIX_DIR}/lib")
  set (${HDF4_PACKAGE_NAME}_JAVA_LIBRARIES "${${HDF4_PACKAGE_NAME}_JAVA_LIBRARY}")
endif()

#-----------------------------------------------------------------------------
# Directories
#-----------------------------------------------------------------------------
set (${HDF4_PACKAGE_NAME}_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include" "${${HDF4_PACKAGE_NAME}_MPI_C_INCLUDE_PATH}" )

set (${HDF4_PACKAGE_NAME}_SHARE_DIR "${PACKAGE_PREFIX_DIR}/cmake")
set_and_check (${HDF4_PACKAGE_NAME}_BUILD_DIR "${PACKAGE_PREFIX_DIR}")

if (${HDF4_PACKAGE_NAME}_BUILD_FORTRAN)
  set (${HDF4_PACKAGE_NAME}_INCLUDE_DIR_FORTRAN "${PACKAGE_PREFIX_DIR}/include" )
endif ()

if (${HDF4_PACKAGE_NAME}_BUILD_TOOLS)
  set (${HDF4_PACKAGE_NAME}_INCLUDE_DIR_TOOLS "${PACKAGE_PREFIX_DIR}/include" )
  set_and_check (${HDF4_PACKAGE_NAME}_TOOLS_DIR "${PACKAGE_PREFIX_DIR}/bin" )
endif ()


if (${HDF4_PACKAGE_NAME}_BUILD_UTILS)
  set (${HDF4_PACKAGE_NAME}_INCLUDE_DIR_UTILS "${PACKAGE_PREFIX_DIR}/include" )
  set_and_check (${HDF4_PACKAGE_NAME}_UTILS_DIR "${PACKAGE_PREFIX_DIR}/bin" )
endif ()

#-----------------------------------------------------------------------------
# Version Strings
#-----------------------------------------------------------------------------
set (${HDF4_PACKAGE_NAME}_VERSION_STRING 4.2.15)
set (${HDF4_PACKAGE_NAME}_VERSION_MAJOR  4.2)
set (${HDF4_PACKAGE_NAME}_VERSION_MINOR  15)

#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already built hdf4 as a subproject
#-----------------------------------------------------------------------------
if (NOT TARGET "hdf4")
  if (${HDF4_PACKAGE_NAME}_ENABLE_JPEG_LIB_SUPPORT AND ${HDF4_PACKAGE_NAME}_PACKAGE_EXTLIBS)
    include (${PACKAGE_PREFIX_DIR}/cmake//-targets.cmake)
  endif ()
  if (${HDF4_PACKAGE_NAME}_ENABLE_Z_LIB_SUPPORT AND ${HDF4_PACKAGE_NAME}_PACKAGE_EXTLIBS)
    include (${PACKAGE_PREFIX_DIR}/cmake//-targets.cmake)
  endif ()
  if (${HDF4_PACKAGE_NAME}_ENABLE_SZIP_SUPPORT AND ${HDF4_PACKAGE_NAME}_PACKAGE_EXTLIBS)
    include (${PACKAGE_PREFIX_DIR}/cmake//-targets.cmake)
  endif ()
  include (${PACKAGE_PREFIX_DIR}/cmake/hdf4/hdf4-targets.cmake)
endif ()

# Handle default component(static) :
if (NOT ${HDF4_PACKAGE_NAME}_FIND_COMPONENTS)
  if (${HDF4_PACKAGE_NAME}_BUILD_STATIC_LIBS)
    set (${HDF4_PACKAGE_NAME}_LIB_TYPE)
    set (${HDF4_PACKAGE_NAME}_FIND_COMPONENTS C static)
    set (${HDF4_PACKAGE_NAME}_FIND_REQUIRED_static_C true)
  else ()
    set (${HDF4_PACKAGE_NAME}_LIB_TYPE)
    set (${HDF4_PACKAGE_NAME}_FIND_COMPONENTS C shared)
    set (${HDF4_PACKAGE_NAME}_FIND_REQUIRED_shared_C true)
  endif ()
endif ()

# Handle requested components:
list (REMOVE_DUPLICATES ${HDF4_PACKAGE_NAME}_FIND_COMPONENTS)
foreach (comp IN LISTS ${HDF4_PACKAGE_NAME}_FIND_COMPONENTS)
  if (comp STREQUAL "shared")
    list (REMOVE_ITEM ${HDF4_PACKAGE_NAME}_FIND_COMPONENTS ${comp})
    set (${HDF4_PACKAGE_NAME}_LIB_TYPE ${${HDF4_PACKAGE_NAME}_LIB_TYPE} ${comp})
  elseif (comp STREQUAL "static")
    list (REMOVE_ITEM ${HDF4_PACKAGE_NAME}_FIND_COMPONENTS ${comp})
    set (${HDF4_PACKAGE_NAME}_LIB_TYPE ${${HDF4_PACKAGE_NAME}_LIB_TYPE} ${comp})
  endif ()
endforeach ()
foreach (libtype IN LISTS ${HDF4_PACKAGE_NAME}_LIB_TYPE)
  foreach (comp IN LISTS ${HDF4_PACKAGE_NAME}_FIND_COMPONENTS)
    set (hdf4_comp2)
    if (comp STREQUAL "C")
      set (hdf4_comp "hdf")
    elseif (comp STREQUAL "Java")
      set (hdf4_comp "hdf_java")
    elseif (comp STREQUAL "Fortran")
      set (hdf4_comp2 "hdf_fcstub")
      set (hdf4_comp "hdf_fortran")
    endif ()
    if (comp STREQUAL "Java")
      list (FIND ${HDF4_PACKAGE_NAME}_EXPORT_LIBRARIES "${hdf4_comp}" HAVE_COMP)
      if (${HAVE_COMP} LESS 0)
        set (${HDF4_PACKAGE_NAME}_${comp}_FOUND 0)
      else ()
        set (${HDF4_PACKAGE_NAME}_${comp}_FOUND 1)
        string(TOUPPER ${HDF4_PACKAGE_NAME}_${comp}_LIBRARY COMP_LIBRARY)
        set (${COMP_LIBRARY} ${${COMP_LIBRARY}} hdf4::${hdf4_comp})
      endif ()
    else ()
      list (FIND ${HDF4_PACKAGE_NAME}_EXPORT_LIBRARIES "${hdf4_comp}-${libtype}" HAVE_COMP)
      list (FIND ${HDF4_PACKAGE_NAME}_EXPORT_LIBRARIES "mf${hdf4_comp}-${libtype}" HAVE_MFCOMP)
      if (${HAVE_COMP} LESS 0 OR ${HAVE_MFCOMP} LESS 0)
        set (${HDF4_PACKAGE_NAME}_${libtype}_${comp}_FOUND 0)
      else ()
        if (hdf4_comp2)
          list (FIND ${HDF4_PACKAGE_NAME}_EXPORT_LIBRARIES "${hdf4_comp2}-${libtype}" HAVE_COMP2)
          list (FIND ${HDF4_PACKAGE_NAME}_EXPORT_LIBRARIES "mf${hdf4_comp2}-${libtype}" HAVE_MFCOMP2)
          if (${HAVE_COMP2} LESS 0 OR ${HAVE_MFCOMP2} LESS 0)
            set (${HDF4_PACKAGE_NAME}_${libtype}_${comp}_FOUND 0)
          else ()
            set (${HDF4_PACKAGE_NAME}_${libtype}_${comp}_FOUND 1)
            string(TOUPPER ${HDF4_PACKAGE_NAME}_${comp}_${libtype}_LIBRARY COMP_LIBRARY)
            set (${COMP_LIBRARY} ${${COMP_LIBRARY}} ${hdf4_comp2}-${libtype} ${hdf4_comp}-${libtype} hdf4::mf${hdf4_comp2}-${libtype} hdf4::mf${hdf4_comp}-${libtype})
          endif ()
        else ()
          set (${HDF4_PACKAGE_NAME}_${libtype}_${comp}_FOUND 1)
          string(TOUPPER ${HDF4_PACKAGE_NAME}_${comp}_${libtype}_LIBRARY COMP_LIBRARY)
          set (${COMP_LIBRARY} ${${COMP_LIBRARY}} ${hdf4_comp}-${libtype} mf${hdf4_comp}-${libtype})
        endif ()
      endif ()
    endif ()
  endforeach ()
endforeach ()

foreach (libtype IN LISTS ${HDF4_PACKAGE_NAME}_LIB_TYPE)
  check_required_components(${HDF4_PACKAGE_NAME}_${libtype})
endforeach ()
