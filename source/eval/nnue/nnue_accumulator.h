// NNUE評価関数の差分計算用のクラス

#ifndef _NNUE_ACCUMULATOR_H_
#define _NNUE_ACCUMULATOR_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_architecture.h"

namespace Eval {

namespace NNUE {

// 入力特徴量をアフィン変換した結果を保持するクラス
// 最終的な出力である評価値も一緒に持たせておく
#if defined(USE_DUAL_NET)
template<Stockfish::Eval::NNUE::IndexType Size>
#endif
struct alignas(32) Accumulator {
#if defined(USE_DUAL_NET)
  std::int16_t
      accumulation[2][kRefreshTriggers.size()][Size];
#else
  std::int16_t
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
#endif
  Value score = VALUE_ZERO;
  bool computed_accumulation = false;
  bool computed_score = false;
};

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)

#endif
