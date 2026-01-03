#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace agafonov_i_sparse_matrix_ccs {

struct SparseMatrixCCS {
  int m = 0;
  int n = 0;
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_ptr;

  SparseMatrixCCS() = default;
  SparseMatrixCCS(int m_val, int n_val) : m(m_val), n(n_val), col_ptr(static_cast<std::size_t>(n_val) + 1, 0) {}
};

struct InType {
  SparseMatrixCCS A;
  SparseMatrixCCS B;
};

using OutType = SparseMatrixCCS;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace agafonov_i_sparse_matrix_ccs
