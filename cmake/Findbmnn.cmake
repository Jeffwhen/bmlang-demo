include(FindPackageHandleStandardArgs)

if (NOT DEFINED ENV{REL_TOP})
    message(FATAL_ERROR "Please source envsetup_xx.sh")
endif()

set(bmlib_header bmlib_runtime.h)
set(bmlang_header bmlang.h)
set(bmrt_header bmruntime_interface.h)
set(bmrt_header_path $ENV{REL_TOP}/include/bmruntime)
set(required_vars)
string(REPLACE ":" ";" ld_path "$ENV{LD_LIBRARY_PATH}")
foreach (lib bmlib bmlang bmrt)
    find_library(${lib}_LIBRARY ${lib} HINTS ${ld_path})
    if (DEFINED ${lib}_header_path)
        set(header_path ${${lib}_header_path})
    else()
        set(header_path $ENV{REL_TOP}/include/${lib})
    endif()
    find_path(${lib}_INCLUDE_DIR ${${lib}_header} HINTS ${header_path})
    list(APPEND required_vars ${lib}_LIBRARY ${lib}_INCLUDE_DIR)

    find_package_handle_standard_args(
        ${lib}
        FOUND_VAR ${lib}_FOUND
        REQUIRED_VARS ${lib}_LIBRARY ${lib}_INCLUDE_DIR)

    if (${lib}_FOUND AND NOT TARGET bmnn::${lib})
        add_library(bmnn::${lib} SHARED IMPORTED)
        set_target_properties(
            bmnn::${lib} PROPERTIES
            IMPORTED_LOCATION "${${lib}_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${${lib}_INCLUDE_DIR}")
    endif()
endforeach()

find_package_handle_standard_args(
    bmnn
    FOUND_VAR bmnn_FOUND
    REQUIRED_VARS ${required_vars})
