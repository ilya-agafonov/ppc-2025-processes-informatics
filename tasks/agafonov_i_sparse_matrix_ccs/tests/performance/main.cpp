#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "agafonov_i_sparse_matrix_ccs/common/include/common.hpp"
#include "agafonov_i_sparse_matrix_ccs/mpi/include/ops_mpi.hpp"
#include "agafonov_i_sparse_matrix_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace agafonov_i_sparse_matrix_ccs {

static SparseMatrixCCS CreatePerfMatrix(int m, int n, double density) {
  SparseMatrixCCS matrix(m, n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_real_distribution<> val_dis(-100.0, 100.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      if (dis(gen) < density) {
        matrix.values.push_back(val_dis(gen));
        matrix.row_indices.push_back(i);
      }
    }
    matrix.col_ptr[static_cast<std::size_t>(j) + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}

class SparseMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_size = 4000;
  void SetUp() override {
    const int m = k_size;
    const int k = k_size;
    const int n = k_size;
    const double density = 0.01;

    input_data_.A = CreatePerfMatrix(m, k, density);
    input_data_.B = CreatePerfMatrix(k, n, density);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.m == k_size && output_data.n == k_size;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(SparseMatrixPerfTests, RunPerfModes) {
  const auto &params = GetParam();
  ExecuteTest(params);
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SparseMatrixCCSResMPI, SparseMatrixCCSSeq>(
    PPC_SETTINGS_agafonov_i_sparse_matrix_ccs);

INSTANTIATE_TEST_SUITE_P(RunModeTests, SparseMatrixPerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         [](const testing::TestParamInfo<SparseMatrixPerfTests::ParamType> &info) {
                           const std::string &name = std::get<1>(info.param);
                           return name + "_" + std::to_string(info.index);
                         });

}  // namespace agafonov_i_sparse_matrix_ccs
