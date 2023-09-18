// NNUE評価関数の入力特徴量HalfKAの定義

#include "../../../config.h"

#if defined(EVAL_NNUE)

#include "half_kae.h"
#include "index_list.h"

namespace Eval {

namespace NNUE {

namespace Features {

namespace {
inline Square GetSquareFromBonaPiece(BonaPiece p) {
  if (p < fe_hand_end) {
    return SQ_NB;
  }
  return static_cast<Square>((p - fe_hand_end) % SQ_NB);
}

inline int GetEffectCount(const Position& pos, Square sq_p,
  Color perspective, bool prev_effect) {
  if (sq_p == SQ_NB) {
    return 0;
  }
  if (perspective == WHITE) {
    sq_p = Inv(sq_p);
  }
  const auto& board_effect = prev_effect ? pos.board_effect_prev : pos.board_effect;
  auto effect1 = board_effect[perspective].effect(sq_p) > 0 ? 1 : 0;
  auto effect2 = board_effect[~perspective].effect(sq_p) > 0 ? 1 : 0;
  return effect1 << 1 | effect2;
}

inline bool IsDirty(const Eval::DirtyPiece& dp, PieceNumber pn) {
  for (int i = 0; i < dp.dirty_num; ++i) {
    if (pn == dp.pieceNo[i]) {
      return true;
    }
  }
  return false;
}
}

// 玉の位置とBonaPieceから特徴量のインデックスを求める
template <Side AssociatedKing>
inline IndexType HalfKAE<AssociatedKing>::MakeIndex(Square sq_k, BonaPiece p, int effect) {
  return static_cast<IndexType>(fe_end2) * static_cast<IndexType>(sq_k) + p +
    static_cast<int>(Eval::fe_end2 - Eval::fe_hand_end) * static_cast<int>(SQ_NB) * effect;
}

// 駒の情報を取得する
template <Side AssociatedKing>
inline void HalfKAE<AssociatedKing>::GetPieces(
    const Position& pos, Color perspective,
    BonaPiece** pieces, Square* sq_target_k) {
  *pieces = (perspective == BLACK) ?
      pos.eval_list()->piece_list_fb() :
      pos.eval_list()->piece_list_fw();
  const PieceNumber target = (AssociatedKing == Side::kFriend) ?
      static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective) :
      static_cast<PieceNumber>(PIECE_NUMBER_KING + ~perspective);
  *sq_target_k = static_cast<Square>(((*pieces)[target] - f_king) % SQ_NB);
}

// 特徴量のうち、値が1であるインデックスのリストを取得する
template <Side AssociatedKing>
void HalfKAE<AssociatedKing>::AppendActiveIndices(
    const Position& pos, Color perspective, IndexList* active) {
  // コンパイラの警告を回避するため、配列サイズが小さい場合は何もしない
  if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
    BonaPiece p = pieces[i];
    Square sq_p = GetSquareFromBonaPiece(p);

    active->push_back(MakeIndex(sq_target_k, p,
      GetEffectCount(pos, sq_p, perspective, false)));
  }
}

// 特徴量のうち、一手前から値が変化したインデックスのリストを取得する
template <Side AssociatedKing>
void HalfKAE<AssociatedKing>::AppendChangedIndices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added) {
  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  const auto& dp = pos.state()->dirtyPiece;
  for (int i = 0; i < dp.dirty_num; ++i) {
    const auto old_p = static_cast<BonaPiece>(
        dp.changed_piece[i].old_piece.from[perspective]);
    auto old_sq_p = GetSquareFromBonaPiece(old_p);
    removed->push_back(MakeIndex(sq_target_k, old_p, GetEffectCount(pos, old_sq_p, perspective, true)));
    const auto new_p = static_cast<BonaPiece>(
        dp.changed_piece[i].new_piece.from[perspective]);
    auto new_sq_p = GetSquareFromBonaPiece(new_p);
    added->push_back(MakeIndex(sq_target_k, new_p, GetEffectCount(pos, new_sq_p, perspective, true)));
  }

  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_NB; ++i) {
    if (IsDirty(dp, i)) {
      continue;
    }

    BonaPiece p = pieces[i];
    Square sq_p = GetSquareFromBonaPiece(p);

    auto effect_prev = GetEffectCount(pos, sq_p, perspective, true);
    auto effect_now = GetEffectCount(pos, sq_p, perspective, false);

    if (effect_prev != effect_now) {
      removed->push_back(MakeIndex(sq_target_k, p, effect_prev));
      added->push_back(MakeIndex(sq_target_k, p, effect_now));
    }
  }
}

template class HalfKAE<Side::kFriend>;
template class HalfKAE<Side::kEnemy>;

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)
