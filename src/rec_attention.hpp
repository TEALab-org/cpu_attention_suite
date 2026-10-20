#pragma once

#include <cilk/cilk_api.h>
#include <utility>

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

class Range {
public:
  Eigen::Index start;
  Eigen::Index end;

  Range(Eigen::Index _start, Eigen::Index _end): start(_start), end(_end) {}

  std::pair<Range, Range> split() {
    auto diff = (end - start) / 2;
    
    auto r1 = Range(start, start + diff);
    auto r2 = Range(start + diff, end);
    return std::make_pair(r1, r2); 
  }

  const Eigen::Index size() {
    end - start;
  }
};

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

  // TODO: make thread local
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P;

  // Result
  AttentionMatrixf O;

  // Base case size
  Eigen::Index base;

  RecAttentionUtil(AttentionProblem const& _problem, Eigen::Index _base): 
    problem(_problem),
    base(_base)
  {
    const Eigen::Index N = problem.Q.rows();
    m = RowVector::Constat(N, -std::numeric_limits<float>::infinity());
    l = RowVector::Zero(N);
    u = RowVector::Zero(N);

    auto const nworkers = __cilkrts_get_nworkers();
    m_t = RowVectorScratch(N, nworkers);
    l_t = RowVectorScratch(N, nworkers);
    u_t = RowVectorScartch(N, nworkers);

    S = Eigen::Matrix(64, 64);
    P = Eigen::Matrix(64, 64);

    O = AttentionMatrixf(N, 64);
  }

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

  void dense_attention() {
    // Already initialized m, l, u, and O
    
    const Eigen::Index N = problem.Q.rows();
    auto r1 = Range(0, N);
    auto r2 = Range(0, N);
    dense_accumulate(r1, r2); 

    // TODO construct O 
  }

  void dense_accumulate(Range range_i, Range range_j) {
    if (range_i.size() <= base || range_j.size() <= base) {
        


      return;
    }


    auto const i_ranges = range_i.split();
    auto const j_ranges = range_j.split();
    dense_accumulate(i_ranges.0, j_ranges.0);
    dense_accumulate(i_ranges.0, j_ranges.1);
    dense_accumulate(i_ranges.1, j_ranges.0);
    dense_accumulate(i_ranges.1, j_ranges.0);
  }



};

/*
void dense_attention_rec(
    // range I,
    // range J,
    // m, l, u
*/


