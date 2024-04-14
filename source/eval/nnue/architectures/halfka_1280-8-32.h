#ifndef NNUE_HALFKA_1280_8_32_H_INCLUDED
#define NNUE_HALFKA_1280_8_32_H_INCLUDED

#include <memory>

#include "../stockfish_nnue/nnue_common.h"

#include "../features/feature_set.h"
#include "../features/half_ka.h"

#include "../stockfish_nnue/layers/affine_transform.h"
#include "../stockfish_nnue/layers/clipped_relu.h"
#include "../stockfish_nnue/layers/sqr_clipped_relu.h"

#include "../../../misc.h"

namespace Eval::NNUE {

// Input features used in evaluation function
using RawFeatures = Features::FeatureSet<
    Features::HalfKA<Features::Side::kFriend>>;

// Number of input feature dimensions after conversion
constexpr Stockfish::Eval::NNUE::IndexType kTransformedFeatureDimensions = 1280;

struct Network
{
  static constexpr int FC_0_OUTPUTS = 8;
  static constexpr int FC_1_OUTPUTS = 32;
  static constexpr int kOutputDimensions = 1;
  using OutputType = std::int32_t;

  Stockfish::Eval::NNUE::Layers::AffineTransform<kTransformedFeatureDimensions, FC_0_OUTPUTS> fc_0;
  Stockfish::Eval::NNUE::Layers::ClippedReLU<FC_0_OUTPUTS> ac_0;
  Stockfish::Eval::NNUE::Layers::SqrClippedReLU<FC_0_OUTPUTS> ac_sqr_0;
  Stockfish::Eval::NNUE::Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS> fc_1;
  Stockfish::Eval::NNUE::Layers::ClippedReLU<FC_1_OUTPUTS> ac_1;
  Stockfish::Eval::NNUE::Layers::AffineTransform<FC_1_OUTPUTS, 1> fc_2;

  // Hash value embedded in the evaluation file
  static constexpr std::uint32_t GetHashValue() {
    // input slice hash
    std::uint32_t hashValue = 0xEC42E90Du;
    hashValue ^= kTransformedFeatureDimensions * 2;

    hashValue = decltype(fc_0)::get_hash_value(hashValue);
    hashValue = decltype(ac_0)::get_hash_value(hashValue);
    hashValue = decltype(fc_1)::get_hash_value(hashValue);
    hashValue = decltype(ac_1)::get_hash_value(hashValue);
    hashValue = decltype(fc_2)::get_hash_value(hashValue);

    return hashValue;
  }

  static std::string GetStructureString() { return ""; }

  // Read network parameters
  bool ReadParameters(std::istream& stream) {
    return   fc_0.read_parameters(stream)
          && ac_0.read_parameters(stream)
          && fc_1.read_parameters(stream)
          && ac_1.read_parameters(stream)
          && fc_2.read_parameters(stream);
  }

  // Write network parameters
  bool WriteParameters(std::ostream& stream) const {
    return   fc_0.write_parameters(stream)
          && ac_0.write_parameters(stream)
          && fc_1.write_parameters(stream)
          && ac_1.write_parameters(stream)
          && fc_2.write_parameters(stream);
  }

  std::int32_t Propagate(const TransformedFeatureType* transformedFeatures)
  {
    struct alignas(Stockfish::Eval::NNUE::CacheLineSize) Buffer
    {
      alignas(Stockfish::Eval::NNUE::CacheLineSize) decltype(fc_0)::OutputBuffer fc_0_out;
      alignas(Stockfish::Eval::NNUE::CacheLineSize) decltype(ac_sqr_0)::OutputType ac_sqr_0_out[Stockfish::Eval::NNUE::ceil_to_multiple<IndexType>(FC_0_OUTPUTS * 2, 32)];
      alignas(Stockfish::Eval::NNUE::CacheLineSize) decltype(ac_0)::OutputBuffer ac_0_out;
      alignas(Stockfish::Eval::NNUE::CacheLineSize) decltype(fc_1)::OutputBuffer fc_1_out;
      alignas(Stockfish::Eval::NNUE::CacheLineSize) decltype(ac_1)::OutputBuffer ac_1_out;
      alignas(Stockfish::Eval::NNUE::CacheLineSize) decltype(fc_2)::OutputBuffer fc_2_out;

      Buffer()
      {
          std::memset(this, 0, sizeof(*this));
      }
    };

#if defined(__clang__) && (__APPLE__)
    // workaround for a bug reported with xcode 12
    static thread_local auto tlsBuffer = std::make_unique<Buffer>();
    // Access TLS only once, cache result.
    Buffer& buffer = *tlsBuffer;
#else
    alignas(Stockfish::Eval::NNUE::CacheLineSize) static thread_local Buffer buffer;
#endif

    fc_0.propagate(transformedFeatures, buffer.fc_0_out);
    ac_sqr_0.propagate(buffer.fc_0_out, buffer.ac_sqr_0_out);
    ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
    std::memcpy(buffer.ac_sqr_0_out + FC_0_OUTPUTS, buffer.ac_0_out, FC_0_OUTPUTS * sizeof(decltype(ac_0)::OutputType));
    fc_1.propagate(buffer.ac_sqr_0_out, buffer.fc_1_out);
    ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);
    fc_2.propagate(buffer.ac_1_out, buffer.fc_2_out);

    std::int32_t outputValue = buffer.fc_2_out[0];

    return outputValue;
  }
};

}  // namespace Eval::NNUE

#endif // #ifndef NNUE_HALFKA_1280_8_32_H_INCLUDED
