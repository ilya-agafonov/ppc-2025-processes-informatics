#pragma once

#include "agafonov_i_sparse_matrix_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace agafonov_i_sparse_matrix_ccs {

class SparseMatrixCCSSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SparseMatrixCCSSeq(const InType &in);
  static SparseMatrixCCS Transpose(const SparseMatrixCCS &matrix);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace agafonov_i_sparse_matrix_ccs
