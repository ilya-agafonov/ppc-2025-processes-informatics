#include "agafonov_i_sparse_matrix_ccs/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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

bool SparseMatrixCCSResMPI::RunImpl() {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &A = GetInput().A;
  auto &B = GetInput().B;
  SparseMatrixCCS AT;
  int dims[4];  // m_A, n_A, m_B, n_B

  if (rank == 0) {
    dims[0] = A.m;
    dims[1] = A.n;
    dims[2] = B.m;
    dims[3] = B.n;
    AT = SparseMatrixCCSSeq::Transpose(A);
  }

  MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    AT.m = dims[1];
    AT.n = dims[0];
    B.m = dims[2];
    B.n = dims[3];
  }

  auto bcast_sparse = [&](SparseMatrixCCS &m) {
    int nnz = (rank == 0) ? (int)m.values.size() : 0;
    int cols = (rank == 0) ? m.n : 0;
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      m.n = cols;
      m.values.resize(nnz);
      m.row_indices.resize(nnz);
      m.col_ptr.resize(m.n + 1);
    }
    if (nnz > 0) {
      MPI_Bcast(m.values.data(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(m.row_indices.data(), nnz, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(m.col_ptr.data(), m.n + 1, MPI_INT, 0, MPI_COMM_WORLD);
  };

  bcast_sparse(AT);

  int chunk = B.n / size;
  int remainder = B.n % size;
  std::vector<int> send_counts(size), displs(size);
  for (int i = 0; i < size; ++i) {
    send_counts[i] = chunk + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  int local_n = send_counts[rank];
  SparseMatrixCCS local_B(B.m, local_n);

  if (rank == 0) {
    for (int i = 1; i < size; i++) {
      int start = displs[i];
      int count = send_counts[i];
      int nnz_s = B.col_ptr[start];
      int nnz_c = B.col_ptr[start + count] - nnz_s;
      MPI_Send(&nnz_c, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      if (nnz_c > 0) {
        MPI_Send(&B.values[nnz_s], nnz_c, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        MPI_Send(&B.row_indices[nnz_s], nnz_c, MPI_INT, i, 2, MPI_COMM_WORLD);
      }
      std::vector<int> adj_ptr(count + 1);
      for (int k = 0; k <= count; ++k) {
        adj_ptr[k] = B.col_ptr[start + k] - nnz_s;
      }
      MPI_Send(adj_ptr.data(), count + 1, MPI_INT, i, 3, MPI_COMM_WORLD);
    }
    int r_nnz = B.col_ptr[local_n];
    local_B.values.assign(B.values.begin(), B.values.begin() + r_nnz);
    local_B.row_indices.assign(B.row_indices.begin(), B.row_indices.begin() + r_nnz);
    local_B.col_ptr.assign(B.col_ptr.begin(), B.col_ptr.begin() + local_n + 1);
  } else {
    int l_nnz;
    MPI_Recv(&l_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_B.values.resize(l_nnz);
    local_B.row_indices.resize(l_nnz);
    local_B.col_ptr.resize(local_n + 1);
    if (l_nnz > 0) {
      MPI_Recv(local_B.values.data(), l_nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(local_B.row_indices.data(), l_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Recv(local_B.col_ptr.data(), local_n + 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  SparseMatrixCCS local_C(dims[0], local_n);
  std::vector<double> dense_col(dims[0], 0.0);
  for (int j = 0; j < local_n; ++j) {
    for (int k_ptr = local_B.col_ptr[j]; k_ptr < local_B.col_ptr[j + 1]; ++k_ptr) {
      int k = local_B.row_indices[k_ptr];
      double v = local_B.values[k_ptr];
      if (k < (int)AT.col_ptr.size() - 1) {
        for (int i_p = AT.col_ptr[k]; i_p < AT.col_ptr[k + 1]; ++i_p) {
          dense_col[AT.row_indices[i_p]] += AT.values[i_p] * v;
        }
      }
    }
    for (int i = 0; i < dims[0]; ++i) {
      if (std::abs(dense_col[i]) > 1e-15) {
        local_C.values.push_back(dense_col[i]);
        local_C.row_indices.push_back(i);
        dense_col[i] = 0.0;
      }
    }
    local_C.col_ptr[j + 1] = (int)local_C.values.size();
  }

  if (rank == 0) {
    SparseMatrixCCS &FC = GetOutput();
    FC.m = dims[0];
    FC.n = dims[3];
    FC.values = local_C.values;
    FC.row_indices = local_C.row_indices;
    FC.col_ptr = local_C.col_ptr;

    for (int i = 1; i < size; ++i) {
      int r_nnz;
      MPI_Recv(&r_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<double> rv(r_nnz);
      std::vector<int> rr(r_nnz);
      int r_cols = send_counts[i];
      std::vector<int> rp(r_cols + 1);

      if (r_nnz > 0) {
        MPI_Recv(rv.data(), r_nnz, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(rr.data(), r_nnz, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      MPI_Recv(rp.data(), r_cols + 1, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int current_total_nnz = (int)FC.values.size();
      FC.values.insert(FC.values.end(), rv.begin(), rv.end());
      FC.row_indices.insert(FC.row_indices.end(), rr.begin(), rr.end());
      for (int k = 1; k <= r_cols; ++k) {
        FC.col_ptr.push_back(rp[k] + current_total_nnz);
      }
    }
  } else {
    int l_nnz = (int)local_C.values.size();
    MPI_Send(&l_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (l_nnz > 0) {
      MPI_Send(local_C.values.data(), l_nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      MPI_Send(local_C.row_indices.data(), l_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Send(local_C.col_ptr.data(), local_n + 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
  }

  MPI_Bcast(&GetOutput().m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetOutput().n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SparseMatrixCCSResMPI::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_sparse_matrix_ccs
