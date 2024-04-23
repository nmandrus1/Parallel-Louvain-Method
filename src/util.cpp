#include "util.h"
#include <mpi.h>

ProcInfo::ProcInfo() {

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int width = sqrt(comm_size);

  if(width * width != comm_size) {
    if(rank == 0)
      std::cout << "ERROR: value of " << comm_size << " is not square and therefore cannot be partitoned properly. Aborting..." << std::endl;

    exit(EXIT_FAILURE);
  }


  this->rank = rank;
  this->i = rank % width;
  this->j = std::floor(rank/width);
  this->comm_size = comm_size;
  this->width = width;

  // ranks for MPI Row and Column groups
  int row_ranks[width];
  int col_ranks[width];

  for (int w =0; w < width; w++) {
    // if you're process 4 in a 3x3 grid (middle grid) then 
    // row ranks should be 3, 4, 5
    // col ranks should be 1, 4, 7
    row_ranks[w] = (this->j * width) + w;
    col_ranks[w] = this->i + (w * width);
  }

  MPI_Group world_group, row_group, col_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  MPI_Group_incl(world_group, width, row_ranks, &row_group);
  MPI_Group_incl(world_group, width, col_ranks, &col_group);

  MPI_Comm_create(MPI_COMM_WORLD, row_group, &this->row_comm);
  MPI_Comm_create(MPI_COMM_WORLD, col_group, &this->col_comm);

  MPI_Comm_rank(this->row_comm, &this->row_rank);
  MPI_Comm_rank(this->col_comm, &this->col_rank);
}
