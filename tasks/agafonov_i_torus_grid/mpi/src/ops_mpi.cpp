#include "agafonov_i_torus_grid/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>

#include "agafonov_i_torus_grid/common/include/common.hpp"

namespace agafonov_i_torus_grid {

TorusGridTaskMPI::TorusGridTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TorusGridTaskMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto data = GetInput();

  return (world_size >= 1 && data.source_rank >= 0 && data.source_rank < world_size && data.dest_rank >= 0 &&
          data.dest_rank < world_size);
}

bool TorusGridTaskMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TorusGridTaskMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto data = GetInput();
  int res = 0;

  std::array<int, 2> dims = {0, 0};
  MPI_Dims_create(world_size, 2, dims.data());
  std::array<int, 2> periods = {1, 1};
  MPI_Comm torus_comm = MPI_COMM_NULL;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 0, &torus_comm);

  if (data.source_rank == data.dest_rank) {
    if (world_rank == data.source_rank) {
      res = data.value;
    }
  } else {
    std::array<int, 2> src_coords = {0, 0};
    std::array<int, 2> dest_coords = {0, 0};
    MPI_Cart_coords(torus_comm, data.source_rank, 2, src_coords.data());
    MPI_Cart_coords(torus_comm, data.dest_rank, 2, dest_coords.data());

    int src_cart_rank = 0;
    int dest_cart_rank = 0;
    MPI_Cart_rank(torus_comm, src_coords.data(), &src_cart_rank);
    MPI_Cart_rank(torus_comm, dest_coords.data(), &dest_cart_rank);

    if (world_rank == src_cart_rank) {
      res = data.value;
      MPI_Send(&res, 1, MPI_INT, dest_cart_rank, 0, torus_comm);
    } else if (world_rank == dest_cart_rank) {
      MPI_Recv(&res, 1, MPI_INT, src_cart_rank, 0, torus_comm, MPI_STATUS_IGNORE);
    }
  }

  MPI_Bcast(&res, 1, MPI_INT, data.dest_rank, MPI_COMM_WORLD);
  GetOutput() = res;

  if (torus_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&torus_comm);
  }
  return true;
}

bool TorusGridTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_torus_grid
