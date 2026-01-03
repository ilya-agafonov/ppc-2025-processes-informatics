#include "agafonov_i_sparse_matrix_ccs/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "agafonov_i_sparse_matrix_ccs/common/include/common.hpp"

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

SparseMatrixCCS SparseMatrixCCSSeq::Transpose(const SparseMatrixCCS &matrix) {
  int target_cols = matrix.m;
  int target_rows = matrix.n;
  SparseMatrixCCS at(target_rows, target_cols);

  at.col_ptr.assign(static_cast<std::size_t>(target_cols) + 1, 0);
  for (int row_indice : matrix.row_indices) {
    at.col_ptr[static_cast<std::size_t>(row_indice) + 1]++;
  }

  for (int i = 0; i < target_cols; ++i) {
    at.col_ptr[static_cast<std::size_t>(i) + 1] += at.col_ptr[static_cast<std::size_t>(i)];
  }

  at.row_indices.resize(matrix.values.size());
  at.values.resize(matrix.values.size());
  std::vector<int> current_pos = at.col_ptr;

  for (int col = 0; col < matrix.n; ++col) {
    for (int j = matrix.col_ptr[static_cast<std::size_t>(col)]; j < matrix.col_ptr[static_cast<std::size_t>(col) + 1];
         ++j) {
      int row = matrix.row_indices[static_cast<std::size_t>(j)];
      int dest_pos = current_pos[static_cast<std::size_t>(row)]++;
      at.row_indices[static_cast<std::size_t>(dest_pos)] = col;
      at.values[static_cast<std::size_t>(dest_pos)] = matrix.values[static_cast<std::size_t>(j)];
    }
  }
  return at;
}

bool SparseMatrixCCSSeq::RunImpl() {
  auto &a = GetInput().A;
  auto &b = GetInput().B;
  SparseMatrixCCS at = Transpose(a);

  auto &c = GetOutput();
  c.m = a.m;
  c.n = b.n;
  c.col_ptr.assign(static_cast<std::size_t>(c.n) + 1, 0);
  c.values.clear();
  c.row_indices.clear();

  std::vector<double> dense_col(static_cast<std::size_t>(a.m), 0.0);
  for (int col_b = 0; col_b < b.n; ++col_b) {
    for (int k_ptr = b.col_ptr[static_cast<std::size_t>(col_b)]; k_ptr < b.col_ptr[static_cast<std::size_t>(col_b) + 1];
         ++k_ptr) {
      int k = b.row_indices[static_cast<std::size_t>(k_ptr)];
      double val_b = b.values[static_cast<std::size_t>(k_ptr)];
      if (k < static_cast<int>(at.col_ptr.size()) - 1) {
        for (int i_ptr = at.col_ptr[static_cast<std::size_t>(k)]; i_ptr < at.col_ptr[static_cast<std::size_t>(k) + 1];
             ++i_ptr) {
          dense_col[static_cast<std::size_t>(at.row_indices[static_cast<std::size_t>(i_ptr)])] +=
              at.values[static_cast<std::size_t>(i_ptr)] * val_b;
        }
      }
    }
    for (int i = 0; i < a.m; ++i) {
      if (std::abs(dense_col[static_cast<std::size_t>(i)]) > 1e-15) {
        c.values.push_back(dense_col[static_cast<std::size_t>(i)]);
        c.row_indices.push_back(i);
        dense_col[static_cast<std::size_t>(i)] = 0.0;
      }
    }
    c.col_ptr[static_cast<std::size_t>(col_b) + 1] = static_cast<int>(c.values.size());
  }
  return true;
}

bool SparseMatrixCCSSeq::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_sparse_matrix_ccs
