#pragma once

#include <vector>

#include "agafonov_i_sparse_matrix_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace agafonov_i_sparse_matrix_ccs {

class SparseMatrixCCSResMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SparseMatrixCCSResMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void MultiplyColumn(int j, int dims0, const SparseMatrixCCS &at, const SparseMatrixCCS &local_b,
                             std::vector<double> &dense_col, SparseMatrixCCS &local_c);

  static void BroadcastSparseMatrix(SparseMatrixCCS &m, int rank);
  static void DistributeData(const SparseMatrixCCS &b, SparseMatrixCCS &local_b, int size, int rank,
                             const std::vector<int> &send_counts, const std::vector<int> &displs);
  static void GatherResults(SparseMatrixCCS &local_c, int size, int rank, const std::vector<int> &send_counts,
                            int dims0, int dims3);
};

}  // namespace agafonov_i_sparse_matrix_ccs
