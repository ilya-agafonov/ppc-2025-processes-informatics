#include "agafonov_i_sparse_matrix_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "agafonov_i_sparse_matrix_ccs/common/include/common.hpp"
#include "agafonov_i_sparse_matrix_ccs/seq/include/ops_seq.hpp"

namespace agafonov_i_sparse_matrix_ccs {

SparseMatrixCCSResMPI::SparseMatrixCCSResMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SparseMatrixCCSResMPI::ValidationImpl() {
  return GetInput().A.n == GetInput().B.m;
}

bool SparseMatrixCCSResMPI::PreProcessingImpl() {
  return true;
}

void SparseMatrixCCSResMPI::MultiplyColumn(int j, int dims0, const SparseMatrixCCS &at, const SparseMatrixCCS &local_b,
                                           std::vector<double> &dense_col, SparseMatrixCCS &local_c) {
  for (int k_ptr = local_b.col_ptr[static_cast<size_t>(j)]; k_ptr < local_b.col_ptr[static_cast<size_t>(j) + 1];
       ++k_ptr) {
    int k = local_b.row_indices[static_cast<size_t>(k_ptr)];
    double v = local_b.values[static_cast<size_t>(k_ptr)];
    if (k < static_cast<int>(at.col_ptr.size()) - 1) {
      for (int i_p = at.col_ptr[static_cast<size_t>(k)]; i_p < at.col_ptr[static_cast<size_t>(k) + 1]; ++i_p) {
        dense_col[static_cast<size_t>(at.row_indices[static_cast<size_t>(i_p)])] +=
            at.values[static_cast<size_t>(i_p)] * v;
      }
    }
  }
  for (int i = 0; i < dims0; ++i) {
    if (std::abs(dense_col[static_cast<size_t>(i)]) > 1e-15) {
      local_c.values.push_back(dense_col[static_cast<size_t>(i)]);
      local_c.row_indices.push_back(i);
      dense_col[static_cast<size_t>(i)] = 0.0;
    }
  }
}

void SparseMatrixCCSResMPI::BroadcastSparseMatrix(SparseMatrixCCS &m, int rank) {
  int nnz = (rank == 0) ? static_cast<int>(m.values.size()) : 0;
  int cols = (rank == 0) ? m.n : 0;
  MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    m.n = cols;
    m.values.resize(static_cast<size_t>(nnz));
    m.row_indices.resize(static_cast<size_t>(nnz));
    m.col_ptr.resize(static_cast<size_t>(m.n) + 1);
  }
  if (nnz > 0) {
    MPI_Bcast(m.values.data(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(m.row_indices.data(), nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(m.col_ptr.data(), m.n + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void SparseMatrixCCSResMPI::DistributeData(const SparseMatrixCCS &b, SparseMatrixCCS &local_b, int size, int rank,
                                           const std::vector<int> &send_counts, const std::vector<int> &displs) {
  if (rank == 0) {
    for (int i = 1; i < size; i++) {
      int start = displs[static_cast<size_t>(i)];
      int count = send_counts[static_cast<size_t>(i)];
      int nnz_s = b.col_ptr[static_cast<size_t>(start)];
      int nnz_c = b.col_ptr[static_cast<size_t>(start) + static_cast<size_t>(count)] - nnz_s;
      MPI_Send(&nnz_c, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      if (nnz_c > 0) {
        MPI_Send(&b.values[static_cast<size_t>(nnz_s)], nnz_c, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        MPI_Send(&b.row_indices[static_cast<size_t>(nnz_s)], nnz_c, MPI_INT, i, 2, MPI_COMM_WORLD);
      }
      std::vector<int> adj_ptr(static_cast<size_t>(count) + 1);
      for (int k = 0; k <= count; ++k) {
        adj_ptr[static_cast<size_t>(k)] = b.col_ptr[static_cast<size_t>(start) + k] - nnz_s;
      }
      MPI_Send(adj_ptr.data(), count + 1, MPI_INT, i, 3, MPI_COMM_WORLD);
    }
    int r_nnz = b.col_ptr[static_cast<size_t>(send_counts[0])];
    local_b.values.assign(b.values.begin(), b.values.begin() + r_nnz);
    local_b.row_indices.assign(b.row_indices.begin(), b.row_indices.begin() + r_nnz);
    local_b.col_ptr.assign(b.col_ptr.begin(), b.col_ptr.begin() + send_counts[0] + 1);
  } else {
    int l_nnz = 0;
    MPI_Recv(&l_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_b.values.resize(static_cast<size_t>(l_nnz));
    local_b.row_indices.resize(static_cast<size_t>(l_nnz));
    local_b.col_ptr.resize(static_cast<size_t>(send_counts[static_cast<size_t>(rank)]) + 1);
    if (l_nnz > 0) {
      MPI_Recv(local_b.values.data(), l_nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(local_b.row_indices.data(), l_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Recv(local_b.col_ptr.data(), send_counts[static_cast<size_t>(rank)] + 1, MPI_INT, 0, 3, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void SparseMatrixCCSResMPI::GatherResults(SparseMatrixCCS &local_c, int size, int rank,
                                          const std::vector<int> &send_counts, int dims0, int dims3) {
  if (rank == 0) {
    SparseMatrixCCS &fc = local_c;
    fc.m = dims0;
    fc.n = dims3;
    for (int i = 1; i < size; ++i) {
      int r_nnz = 0;
      MPI_Recv(&r_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<double> rv(static_cast<size_t>(r_nnz));
      std::vector<int> rr(static_cast<size_t>(r_nnz));
      int r_cols = send_counts[static_cast<size_t>(i)];
      std::vector<int> rp(static_cast<size_t>(r_cols) + 1);
      if (r_nnz > 0) {
        MPI_Recv(rv.data(), r_nnz, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(rr.data(), r_nnz, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      MPI_Recv(rp.data(), r_cols + 1, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int current_total_nnz = static_cast<int>(fc.values.size());
      fc.values.insert(fc.values.end(), rv.begin(), rv.end());
      fc.row_indices.insert(fc.row_indices.end(), rr.begin(), rr.end());
      for (int k = 1; k <= r_cols; ++k) {
        fc.col_ptr.push_back(rp[static_cast<size_t>(k)] + current_total_nnz);
      }
    }
  } else {
    int l_nnz = static_cast<int>(local_c.values.size());
    MPI_Send(&l_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (l_nnz > 0) {
      MPI_Send(local_c.values.data(), l_nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      MPI_Send(local_c.row_indices.data(), l_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Send(local_c.col_ptr.data(), static_cast<int>(local_c.col_ptr.size()), MPI_INT, 0, 3, MPI_COMM_WORLD);
  }
}

bool SparseMatrixCCSResMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &a = GetInput().A;
  auto &b = GetInput().B;
  SparseMatrixCCS at;
  std::vector<int> dims(4, 0);

  if (rank == 0) {
    dims[0] = a.m;
    dims[1] = a.n;
    dims[2] = b.m;
    dims[3] = b.n;
    at = SparseMatrixCCSSeq::Transpose(a);
  }
  MPI_Bcast(dims.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    at.m = dims[1];
    at.n = dims[0];
    b.m = dims[2];
    b.n = dims[3];
  }

  BroadcastSparseMatrix(at, rank);

  int chunk = b.n / size;
  int remainder = b.n % size;
  std::vector<int> send_counts(static_cast<size_t>(size));
  std::vector<int> displs(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    send_counts[static_cast<size_t>(i)] = chunk + (i < remainder ? 1 : 0);
    displs[static_cast<size_t>(i)] =
        (i == 0) ? 0 : displs[static_cast<size_t>(i) - 1] + send_counts[static_cast<size_t>(i) - 1];
  }

  SparseMatrixCCS local_b(b.m, send_counts[static_cast<size_t>(rank)]);
  DistributeData(b, local_b, size, rank, send_counts, displs);

  int local_n = send_counts[static_cast<size_t>(rank)];
  SparseMatrixCCS local_c(dims[0], local_n);
  std::vector<double> dense_col(static_cast<size_t>(dims[0]), 0.0);
  for (int j = 0; j < local_n; ++j) {
    MultiplyColumn(j, dims[0], at, local_b, dense_col, local_c);
    local_c.col_ptr[static_cast<size_t>(j) + 1] = static_cast<int>(local_c.values.size());
  }

  if (rank == 0) {
    GatherResults(local_c, size, rank, send_counts, dims[0], dims[3]);
    GetOutput() = local_c;
  } else {
    GatherResults(local_c, size, rank, send_counts, dims[0], dims[3]);
  }

  MPI_Bcast(&GetOutput().m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetOutput().n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool SparseMatrixCCSResMPI::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_sparse_matrix_ccs
