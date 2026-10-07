#pragma once

#include <cilk/cilk_api.h>

inline AttentionMatrixf rec_attention(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V
) {
    const Eigen::Index N = Q.rows();
    //const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));

    AttentionMatrixf O(N, 64);
  
    return O;
}

template <typename Scalar = float>
using RowVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::RowMajor>;

template <typename Scalar = float>
using RowVectorMap = Eigen::Map<RowVector<Scalar>, Eigen::Aligned64>;

template <typename Scalar = float>
using RowVectorScratch = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename Scalar = float>
class RecAttentionUtil {
public:
  AttentionProblem const & problem;

  // Intermediate Buffers
  RowVector<Scalar> m;
  RowVector<Scalar> l;
  RowVector<Scalar> u;

  // Thread local scratch 
  RowVectorScratch<Scalar> m_t;
  RowVectorScratch<Scalar> l_t;
  RowVectorScratch<Scalar> u_t;

  RowVectorMap<Scalar> get_m_scratch() {
    int worker_id = __cilkrts_get_worker_number();
    return m_t.row(worker_id);
  }

  RowVectorMap<Scalar> get_l_scratch() {
    int worker_id = __cilkrts_get_worker_number();
    return l_t.row(worker_id);
  }

  RowVectorMap<Scalar> get_u_scratch() {
    int worker_id = __cilkrts_get_worker_number();
    return u_t.row(worker_id);
  }

};

/*
void dense_attention_rec(
    // range I,
    // range J,
    // m, l, u
*/


