# -*- mode: cmake; -*-
# Time-stamp: <Tue 2015-10-27 14:51 svarrette>
################################################################################
# CMakeLists.txt --  [CMake](http://www.cmake.org/) Configuration
#                 ____ __  __       _        _     _     _
#                / ___|  \/  | __ _| | _____| |   (_)___| |_ ___
#               | |   | |\/| |/ _` | |/ / _ \ |   | / __| __/ __|
#               | |___| |  | | (_| |   <  __/ |___| \__ \ |_\__ \
#                \____|_|  |_|\__,_|_|\_\___|_____|_|___/\__|___/
#
################################################################################
cmake_minimum_required (VERSION 2.8.12)
project (MPI_CircuitSatisfiability C)

SET(CMAKE_C_STANDARD 99)
#SET(CMAKE_C_FLAGS --std=c99 )

enable_testing()

#==== MPI
find_package(MPI REQUIRED)
set_property( DIRECTORY PROPERTY COMPILE_DEFINITIONS ${MPI_C_COMPILE_FLAGS} )
set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} ${MPI_C_COMPILE_FLAGS}")
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
list(APPEND EXTRALIBS ${MPI_C_LIBRARIES})

#==== Let's go
add_executable(circuit       circuit.c)
add_executable(mpi_template  mpi_template.c)
target_link_libraries(mpi_template     ${EXTRALIBS})

add_executable(mpi_circuit_1  mpi_circuit_1.c)
add_executable(mpi_circuit_2  mpi_circuit_2.c)
add_executable(mpi_circuit_3  mpi_circuit_3.c)
target_link_libraries(mpi_circuit_1     ${EXTRALIBS})
target_link_libraries(mpi_circuit_2     ${EXTRALIBS})
target_link_libraries(mpi_circuit_3     ${EXTRALIBS})
#================= Automatic tests ==================
#define a macro to simplify adding tests, then use it
macro (do_test prog result)
  add_test (UniTest-${prog}-contains-${result} ${prog})
  set_tests_properties (UniTest-${prog}-contains-${result}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result})
endmacro (do_test)

set(SOLUTIONS
  1010111110011001
  0110111110011001
  1110111110011001
  1010111111011001
  0110111111011001
  1110111111011001
  1010111110111001
  0110111110111001
  1110111110111001
)

foreach(solution ${SOLUTIONS})
  do_test(circuit ${solution})
endforeach(solution ${SOLUTIONS})

#=== MPI Tests
macro (do_mpi_test prog result)
  add_test (MPI_UniTest-${prog}-expect-${result}
    ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS}
    ${MPIEXEC_PREFLAGS}
    ${prog}
    ${MPIEXEC_POSTFLAGS}
    )
  set_tests_properties (MPI_UniTest-${prog}-expect-${result}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result})
endmacro (do_mpi_test)

# Number of procs for MPI.
set (PROCS 4)

# TODO: find a better version that catch the command output using EXEC_PROGRAM such
# that the program is executed only once. 
foreach(solution ${SOLUTIONS})
#  do_mpi_test(mpi_circuit_1 ${solution})
endforeach(solution ${SOLUTIONS})

do_mpi_test(mpi_circuit_2 "9")

# macro (do_test prog expected_results)
#   EXEC_PROGRAM(${prog}
#     OUTPUT_VARIABLE PROG_OUTPUT)
#   SET(PROG_OUTPUT "${PROG_OUTPUT}" CACHE STRING "${prog} output")
#   message("${prog} output: '${PROG_OUTPUT}'")
#   foreach(sol ${expected_results})
#   add_test(UniTest-${prog}-contains-${sol} echo ${PROG_OUTPUT})
#   set_tests_properties (UniTest-${prog}-contains-${sol}
#     PROPERTIES PASS_REGULAR_EXPRESSION ${sol})
#   endforeach(sol ${expected_results})
# endmacro (do_test)
