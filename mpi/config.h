// Include mpi.h without getting deprecated C++ bindings
#pragma once

#ifdef OMPI_MPI_H
#error "mpi.h" already included
#endif

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
