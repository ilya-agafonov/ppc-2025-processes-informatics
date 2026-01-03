#include "agafonov_i_sparse_matrix_ccs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace agafonov_i_sparse_matrix_ccs {

SparseMatrixCCSSeq::SparseMatrixCCSSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SparseMatrixCCSSeq::ValidationImpl() {
  return GetInput().A.n == GetInput().B.m;
}

bool SparseMatrixCCSSeq::PreProcessingImpl() {
  return true;
}

SparseMatrixCCS SparseMatrixCCSSeq::Transpose(const SparseMatrixCCS &A) {
  int target_cols = A.m;
  int target_rows = A.n;
  SparseMatrixCCS AT(target_rows, target_cols);

  AT.col_ptr.assign(target_cols + 1, 0);
  for (int i = 0; i < (int)A.row_indices.size(); ++i) {
    AT.col_ptr[A.row_indices[i] + 1]++;
  }

  for (int i = 0; i < target_cols; ++i) {
    AT.col_ptr[i + 1] += AT.col_ptr[i];
  }

  AT.row_indices.resize(A.values.size());
  AT.values.resize(A.values.size());
  std::vector<int> current_pos = AT.col_ptr;

  for (int col = 0; col < A.n; ++col) {
    for (int j = A.col_ptr[col]; j < A.col_ptr[col + 1]; ++j) {
      int row = A.row_indices[j];
      int dest_pos = current_pos[row]++;
      AT.row_indices[dest_pos] = col;
      AT.values[dest_pos] = A.values[j];
    }
  }
  return AT;
}

bool SparseMatrixCCSSeq::RunImpl() {
  auto &A = GetInput().A;
  auto &B = GetInput().B;
  SparseMatrixCCS AT = Transpose(A);

  auto &C = GetOutput();
  C.m = A.m;
  C.n = B.n;
  C.col_ptr.assign(C.n + 1, 0);
  C.values.clear();
  C.row_indices.clear();

  std::vector<double> dense_col(A.m, 0.0);
  for (int col_B = 0; col_B < B.n; ++col_B) {
    for (int k_ptr = B.col_ptr[col_B]; k_ptr < B.col_ptr[col_B + 1]; ++k_ptr) {
      int k = B.row_indices[k_ptr];
      double val_B = B.values[k_ptr];
      if (k < (int)AT.col_ptr.size() - 1) {
        for (int i_ptr = AT.col_ptr[k]; i_ptr < AT.col_ptr[k + 1]; ++i_ptr) {
          dense_col[AT.row_indices[i_ptr]] += AT.values[i_ptr] * val_B;
        }
      }
    }
    for (int i = 0; i < A.m; ++i) {
      if (std::abs(dense_col[i]) > 1e-15) {
        C.values.push_back(dense_col[i]);
        C.row_indices.push_back(i);
        dense_col[i] = 0.0;
      }
    }
    C.col_ptr[col_B + 1] = (int)C.values.size();
  }
  return true;
}

bool SparseMatrixCCSSeq::PostProcessingImpl() {
  return true;
}
}  // namespace agafonov_i_sparse_matrix_ccs
