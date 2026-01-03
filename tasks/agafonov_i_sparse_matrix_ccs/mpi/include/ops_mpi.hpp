#pragma once

#include <mpi.h>

#include "agafonov_i_sparse_matrix_ccs/common/include/common.hpp"

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

  void SendSparseMatrix(const SparseMatrixCCS &matrix, int dest, int tag);
  void RecvSparseMatrix(SparseMatrixCCS &matrix, int source, int tag);
};

}  // namespace agafonov_i_sparse_matrix_ccs
