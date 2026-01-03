#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "agafonov_i_sparse_matrix_ccs/common/include/common.hpp"
#include "agafonov_i_sparse_matrix_ccs/mpi/include/ops_mpi.hpp"
#include "agafonov_i_sparse_matrix_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace agafonov_i_sparse_matrix_ccs {

typedef std::tuple<int, int, int, double, std::string> TestParams;

static SparseMatrixCCS CreateRandomSparseMatrix(int m, int n, double density) {
  SparseMatrixCCS matrix(m, n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_real_distribution<> val_dis(-100.0, 100.0);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      if (dis(gen) < density) {
        matrix.values.push_back(val_dis(gen));
        matrix.row_indices.push_back(i);
      }
    }
    matrix.col_ptr[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}

class SparseMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestParams> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<typename SparseMatrixFuncTests::ParamType> &info) {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    std::string test_name = std::get<4>(params);
    return test_name + "_" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int m = std::get<0>(params);
    int k = std::get<1>(params);
    int n = std::get<2>(params);
    double density = std::get<3>(params);

    input_data_.A = CreateRandomSparseMatrix(m, k, density);
    input_data_.B = CreateRandomSparseMatrix(k, n, density);

    SparseMatrixCCSSeq task_seq(input_data_);
    task_seq.Validation();
    task_seq.PreProcessing();
    task_seq.Run();
    task_seq.PostProcessing();
    expected_output_ = task_seq.GetOutput();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }

    if (output_data.values.size() != expected_output_.values.size()) {
      return false;
    }
    if (output_data.col_ptr != expected_output_.col_ptr) {
      return false;
    }
    if (output_data.row_indices != expected_output_.row_indices) {
      return false;
    }

    for (size_t i = 0; i < output_data.values.size(); ++i) {
      if (std::abs(output_data.values[i] - expected_output_.values[i]) > 1e-6) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(SparseMatrixFuncTests, MatmulTests) {
  ExecuteTest(GetParam());
}

namespace {
const std::array<TestParams, 6> kFuncTestParams = {
    std::make_tuple(10, 10, 10, 0.1, "Square_Small"),        std::make_tuple(32, 32, 32, 0.2, "Square_Mid"),
    std::make_tuple(15, 7, 20, 0.15, "Rectangular_Diverse"), std::make_tuple(1, 50, 1, 0.3, "Inner_Product_Vector"),
    std::make_tuple(50, 1, 50, 0.8, "Outer_Product_Dense"),  std::make_tuple(20, 20, 20, 0.0, "Empty_Matrix")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SparseMatrixCCSResMPI, InType>(kFuncTestParams, PPC_SETTINGS_agafonov_i_sparse_matrix_ccs),
    ppc::util::AddFuncTask<SparseMatrixCCSSeq, InType>(kFuncTestParams, PPC_SETTINGS_agafonov_i_sparse_matrix_ccs));

INSTANTIATE_TEST_SUITE_P(SparseMatrixTests, SparseMatrixFuncTests, ppc::util::ExpandToValues(kTestTasksList),
                         SparseMatrixFuncTests::PrintTestParam);
}  // namespace

}  // namespace agafonov_i_sparse_matrix_ccs
