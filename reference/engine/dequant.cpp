#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <cstring>
#include <vector>
extern "C" int flash_memory_dequant(int type_id, const void * input, float * output, int64_t count) {
    if (type_id < 0 || type_id >= GGML_TYPE_COUNT || count < 0) return -1;
    const auto type = static_cast<ggml_type>(type_id);
    if (type == GGML_TYPE_F32) { memcpy(output, input, size_t(count)*sizeof(float)); return 0; }
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float || count % ggml_blck_size(type)) return -2;
    traits->to_float(input, output, count);
    return 0;
}
extern "C" int flash_memory_activation_quant(int weight_type_id, const float * input, float * output, int64_t width, int64_t rows) {
    if (weight_type_id < 0 || weight_type_id >= GGML_TYPE_COUNT || width < 1 || rows < 1) return -1;
    const auto * weight_traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(weight_type_id));
    if (!weight_traits) return -2;
    const auto activation_type = weight_traits->vec_dot_type;
    if (activation_type == GGML_TYPE_F32) { memcpy(output,input,size_t(width*rows)*4);return 0; }
    const auto * cpu = ggml_get_type_traits_cpu(activation_type);
    const auto * generic = ggml_get_type_traits(activation_type);
    if (!cpu || !cpu->from_float || !generic || (!generic->to_float && activation_type!=GGML_TYPE_Q8_K) || width%ggml_blck_size(activation_type)) return -3;
    std::vector<unsigned char> quantized(ggml_row_size(activation_type,width));
    for (int64_t r=0;r<rows;++r) {
        cpu->from_float(input+r*width,quantized.data(),width);
        if (activation_type==GGML_TYPE_Q8_K) {
            dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(quantized.data()),output+r*width,width);
        } else generic->to_float(quantized.data(),output+r*width,width);
    }
    return int(activation_type);
}
