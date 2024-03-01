enable_language(CXX Fortran)

set(SOURCES
  src/lapack_cpp/rot_cpp.cpp
  src/lapack_cpp/lartg_cpp.cpp
  src/lapack_cpp/rot_cpp.cpp
  src/lapack_c/lartg_c.f90
  src/lapack_c/lasr3_c.cpp
  src/lapack_c/rot_c.f90
  src/lapack_fortran/clasr3.f90
  src/lapack_fortran/dlasr3.f90
  src/lapack_fortran/slasr3.f90
  src/lapack_fortran/zlasr3.f90
)

include_directories(include)
add_compile_options(-Wextra -Wall -pedantic)

add_library(lapackcpp ${SOURCES})
target_link_libraries(lapackcpp)
set_target_properties(
  lapackcpp PROPERTIES
  VERSION ${LAPACK_VERSION}
  SOVERSION ${LAPACK_MAJOR_VERSION}
)
