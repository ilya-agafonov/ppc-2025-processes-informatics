#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace agafonov_i_sparse_matrix_ccs {

struct SparseMatrixCCS {
  int m, n;
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_ptr;

  SparseMatrixCCS() : m(0), n(0) {}
  SparseMatrixCCS(int _m, int _n) : m(_m), n(_n) {
    col_ptr.resize(n + 1, 0);
  }
};
struct InType {
  SparseMatrixCCS A;
  SparseMatrixCCS B;
};

using OutType = SparseMatrixCCS;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace agafonov_i_sparse_matrix_ccs
