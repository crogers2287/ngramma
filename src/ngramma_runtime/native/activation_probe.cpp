// Original integration: MIT. Calls externally supplied GGML CPU traits.
// Restricted to ordinary (non-repacked) Q8_0 PLE projections; forward only.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <limits>
#include <mutex>

extern "C" int ngramma_activation_encode_q8_0(int weight_type, const float * input,
        uint8_t * output, int64_t width, int64_t rows, uint64_t output_bytes) {
    if (weight_type != GGML_TYPE_Q8_0) return -1;
    if (!input || !output || width <= 0 || rows <= 0 || width % 32) return -2;
    if (uint64_t(width) > uint64_t(INT64_MAX) / 4 / uint64_t(rows)) return -3;
    static std::once_flag initialized;
    std::call_once(initialized, ggml_cpu_init);
    const auto * weight = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (weight->vec_dot_type != GGML_TYPE_Q8_0) return -4;
    const auto * activation = ggml_get_type_traits_cpu(weight->vec_dot_type);
    if (!activation->from_float || ggml_blck_size(weight->vec_dot_type) != 32 ||
        ggml_type_size(weight->vec_dot_type) != 34) return -4;
    const size_t row_bytes = ggml_row_size(weight->vec_dot_type, width);
    if (uint64_t(row_bytes) > std::numeric_limits<uint64_t>::max() / uint64_t(rows) ||
        output_bytes != uint64_t(row_bytes) * uint64_t(rows)) return -3;
    for (int64_t row = 0; row < rows; ++row)
        activation->from_float(input + row * width, output + row * row_bytes, width);
    return 0;
}
