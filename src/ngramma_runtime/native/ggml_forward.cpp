// Original integration: MIT, see LICENSE.
// Calls separately supplied ggml CPU primitives (ggml authors, MIT;
// see licenses/llama-cpp-MIT.txt). No backward/gradient implementation.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_message;
thread_local std::string selected_buffer_type;
std::once_flag cpu_initialized;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

void require(bool okay, const char * message) {
    if (!okay) throw std::invalid_argument(message);
}

void dimensions(int64_t width, int64_t rows, int threads) {
    require(width > 0 && rows > 0, "Tensor dimensions must be positive");
    require(width <= INT32_MAX && rows <= INT32_MAX, "Dimensions exceed CPU primitive integer range");
    require(uint64_t(width) <= uint64_t(std::numeric_limits<int64_t>::max()) / sizeof(float) / uint64_t(rows), "Tensor byte size overflows");
    require(threads > 0 && threads <= GGML_MAX_N_THREADS, "Thread count outside GGML range");
}

Context context() {
    std::call_once(cpu_initialized, ggml_cpu_init);
    ggml_init_params params{};
    params.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    Context result(ggml_init(params), ggml_free);
    if (!result) throw std::runtime_error("Unable to allocate GGML graph context");
    return result;
}

void compute(ggml_context * ctx, ggml_tensor * result, float * output, int threads) {
    result->data = output;
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    ggml_cplan plan = ggml_graph_plan(graph, threads, nullptr);
    // Extra alignment space is deliberately independent of GGML work_size.
    std::vector<uint8_t> work(plan.work_size ? plan.work_size + GGML_MEM_ALIGN : 0);
    if (plan.work_size) {
        const uintptr_t start = reinterpret_cast<uintptr_t>(work.data());
        plan.work_data = reinterpret_cast<uint8_t *>((start + GGML_MEM_ALIGN - 1) & ~(uintptr_t(GGML_MEM_ALIGN) - 1));
    } else {
        plan.work_data = nullptr;
    }
    const ggml_status status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("GGML CPU graph failed, status " + std::to_string(int(status)));
}

template<class Operation> int guarded(Operation operation) noexcept {
    error_message.clear();
    try { operation(); return 0; }
    catch (const std::exception & error) { error_message = error.what(); return -1; }
    catch (...) { error_message = "Unknown GGML forward bridge exception"; return -1; }
}
} // namespace

extern "C" {
// Message lifetime: until this thread's next bridge invocation.
const char * ngramma_last_error() noexcept { return error_message.c_str(); }
// Empty before a successful repack selection; overwritten each repack call.
const char * ngramma_last_buffer_type() noexcept { return selected_buffer_type.c_str(); }

// All arrays are contiguous and caller-owned for the duration of the call.
// raw_weights: GGML source type, shape [n,k] in row-major storage.
// input: F32 [m,k], output: F32 [m,n]. Output must not alias either input.
// Caller must provide sufficient byte lengths; this C ABI cannot inspect them.
int ngramma_matmul(int type, const void * raw_weights, const float * input,
                   float * output, int64_t k, int64_t n, int64_t m,
                   int threads) noexcept {
    return guarded([&] {
        dimensions(k, n, threads); dimensions(k, m, threads); dimensions(n, m, threads);
        require(raw_weights && input && output, "Null tensor buffer");
        require(type >= 0 && type < GGML_TYPE_COUNT, "Invalid GGML type ID");
        const auto weight_type = static_cast<ggml_type>(type);
        const auto * traits = ggml_get_type_traits_cpu(weight_type);
        require(traits && traits->vec_dot, "GGML type has no supported CPU matrix dot primitive");
        const int64_t block = ggml_blck_size(weight_type);
        require(block > 0 && k % block == 0, "Weight width must be a multiple of its GGML quantization block");
        auto ctx = context();
        ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), weight_type, k, n);
        ggml_tensor * values = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
        weights->data = const_cast<void *>(raw_weights);
        values->data = const_cast<float *>(input);
        compute(ctx.get(), ggml_mul_mat(ctx.get(), weights, values), output, threads);
    });
}


// Optional loader-style CPU extra-buffer experiment. Dimensions and ABI match
// ngramma_matmul; input token rows are never padded or rounded to a tile size.
// Selection reports its actual buffer name through ngramma_last_buffer_type().
// A plain CPU fallback is explicit; extra-buffer availability depends on build
// and hardware. This does not prove that a separately loaded model used it.
int ngramma_matmul_repack(int type, const void * raw_weights, const float * input,
                          float * output, int64_t k, int64_t n, int64_t m,
                          int threads) noexcept {
    selected_buffer_type.clear();
    return guarded([&] {
        dimensions(k, n, threads); dimensions(k, m, threads); dimensions(n, m, threads);
        require(raw_weights && input && output, "Null tensor buffer");
        require(type >= 0 && type < GGML_TYPE_COUNT, "Invalid GGML type ID");
        const auto weight_type = static_cast<ggml_type>(type);
        const auto * traits = ggml_get_type_traits_cpu(weight_type);
        require(traits && traits->vec_dot, "GGML type has no supported CPU matrix dot primitive");
        const int64_t block = ggml_blck_size(weight_type);
        require(block > 0 && k % block == 0, "Weight width must be a multiple of its GGML quantization block");
        auto weight_ctx = context();
        auto graph_ctx = context();
        auto * weights = ggml_new_tensor_2d(weight_ctx.get(), weight_type, k, n);
        auto * values = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, k, m);
        values->data = const_cast<float *>(input);
        auto * result = ggml_mul_mat(graph_ctx.get(), weights, values);
        auto reg = ggml_backend_cpu_reg();
        auto device = ggml_backend_reg_dev_get(reg, 0);
        require(device != nullptr, "CPU backend device unavailable");
        auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
        ggml_backend_buffer_type_t selected = nullptr;
        using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
        if (get_extra) {
            auto extras = get_extra(device);
            for (size_t i = 0; extras && extras[i]; ++i) {
                Buffer dummy(ggml_backend_buft_alloc_buffer(extras[i], 0), ggml_backend_buffer_free);
                if (!dummy) continue;
                weights->buffer = dummy.get();
                const bool supported = ggml_backend_dev_supports_op(device, result);
                weights->buffer = nullptr;
                if (supported) { selected = extras[i]; break; }
            }
        }
        if (!selected) selected = ggml_backend_cpu_buffer_type();
        Buffer buffer(ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx.get(), selected), ggml_backend_buffer_free);
        require(buffer != nullptr, "Unable to allocate selected CPU weight buffer");
        require(ggml_backend_dev_supports_op(device, result), "Selected CPU weight buffer does not support matrix operation");
        ggml_backend_tensor_set(weights, raw_weights, 0, ggml_nbytes(weights));
        selected_buffer_type = ggml_backend_buft_name(selected);
        compute(graph_ctx.get(), result, output, threads);
    });
}

// op: 0 RMSNorm, 1 SiLU, 2 sigmoid, 3 L2Norm, 4 softmax, 5 exp, 6 softplus.
// Normalization/softmax run independently over each width-sized row.
int ngramma_unary(int op, const float * input, float * output,
                  int64_t width, int64_t rows, float eps, int threads) noexcept {
    return guarded([&] {
        dimensions(width, rows, threads);
        require(input && output, "Null tensor buffer");
        require(op >= 0 && op <= 6, "Unknown unary operation ID");
        require(std::isfinite(eps) && eps >= 0, "Epsilon must be finite and nonnegative");
        auto ctx = context();
        ggml_tensor * values = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
        values->data = const_cast<float *>(input);
        ggml_tensor * result = nullptr;
        switch (op) {
            case 0: result = ggml_rms_norm(ctx.get(), values, eps); break;
            case 1: result = ggml_silu(ctx.get(), values); break;
            case 2: result = ggml_sigmoid(ctx.get(), values); break;
            case 3: result = ggml_l2_norm(ctx.get(), values, eps); break;
            case 4: result = ggml_soft_max(ctx.get(), values); break;
            case 5: result = ggml_exp(ctx.get(), values); break;
            case 6: result = ggml_softplus(ctx.get(), values); break;
        }
        compute(ctx.get(), result, output, threads);
    });
}


// Sum float32 rows using the engine's accumulation precision and final cast.
// Input [rows,width], output [rows].
int ngramma_sum_rows(const float * input, float * output, int64_t width,
                      int64_t rows, int threads) noexcept {
    return guarded([&] {
        dimensions(width, rows, threads);
        require(input && output, "Null row-sum tensor buffer");
        auto ctx = context();
        auto * values = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
        values->data = const_cast<float *>(input);
        compute(ctx.get(), ggml_sum_rows(ctx.get(), values), output, threads);
    });
}

// Depthwise causal window primitive, dilation=1, one sequence. The caller
// prepends its own K-1 history values. F32 arrays: input[C,T+K-1],
// kernel[C,K], output[T,C]. No implicit padding, reversal, bias, or activation.
int ngramma_ssm_conv(const float * full_input_ct, const float * kernel_ck,
                     float * output_tc, int64_t tokens, int64_t channels,
                     int64_t kernel, int threads) noexcept {
    return guarded([&] {
        dimensions(tokens, channels, threads);
        dimensions(kernel, channels, threads);
        require(tokens <= INT32_MAX - kernel + 1, "SSM history width exceeds CPU integer range");
        dimensions(tokens + kernel - 1, channels, threads);
        require(full_input_ct && kernel_ck && output_tc, "Null SSM tensor buffer");
        auto ctx = context();
        auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, tokens + kernel - 1, channels, 1);
        auto * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kernel, channels);
        input->data = const_cast<float *>(full_input_ct);
        weights->data = const_cast<float *>(kernel_ck);
        compute(ctx.get(), ggml_ssm_conv(ctx.get(), input, weights), output_tc, threads);
    });
}

// Fused Gated DeltaNet, scalar log-decay g, already-sigmoided beta, unscaled q.
// One sequence, final state only. Caller arrays are contiguous F32:
// q,k[T,Hk,D]; v/output[T,Hv,D]; g,beta[T,Hv]; state/new_state[Hv,D_value,D_key].
// Engine stores conceptual state[key,value] transposed, key index contiguous.
// Hv % Hk == 0; the native broadcast is tiled: q_head = value_head % Hk.
// Native kernel applies exp(g) and scales each state-dot-q by 1/sqrt(D).
// The output/state destinations must not overlap each other. Initial state is
// read-only; output may reuse input storage because copies occur after compute.
int ngramma_gdn(const float * q, const float * k, const float * v,
                const float * g, const float * beta, const float * state,
                float * output, float * new_state, int64_t tokens,
                int64_t key_heads, int64_t value_heads, int64_t dim,
                int threads) noexcept {
    return guarded([&] {
        dimensions(tokens, value_heads, threads);
        dimensions(dim, key_heads, threads);
        dimensions(dim, value_heads, threads);
        dimensions(dim, dim, threads);
        require(value_heads % key_heads == 0, "GDN value_heads must be divisible by key_heads");
        require(dim <= INT32_MAX / dim, "GDN state row exceeds CPU vector integer range");
        // Check the full packed allocation before any shape multiplication.
        const uint64_t max_floats = uint64_t(std::numeric_limits<int64_t>::max()) / sizeof(float);
        const uint64_t per_head = uint64_t(dim) * (uint64_t(tokens) + uint64_t(dim));
        require(uint64_t(value_heads) <= max_floats / per_head, "GDN packed output byte size overflows");
        require(q && k && v && g && beta && state && output && new_state, "Null GDN tensor buffer");
        const size_t output_count = size_t(tokens) * size_t(value_heads) * size_t(dim);
        const size_t state_count = size_t(value_heads) * size_t(dim) * size_t(dim);
        const uintptr_t output_start = reinterpret_cast<uintptr_t>(output);
        const uintptr_t state_start = reinterpret_cast<uintptr_t>(new_state);
        require(output_start <= UINTPTR_MAX - output_count * sizeof(float) &&
                state_start <= UINTPTR_MAX - state_count * sizeof(float), "GDN output pointer range overflows");
        require(output_start + output_count * sizeof(float) <= state_start ||
                state_start + state_count * sizeof(float) <= output_start, "GDN output and new_state buffers must not overlap");
        auto ctx = context();
        auto tensor = [&](const float * data, int64_t a, int64_t b, int64_t c) {
            auto * result = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, a, b, c, 1);
            result->data = const_cast<float *>(data);
            return result;
        };
        auto * tq = tensor(q, dim, key_heads, tokens);
        auto * tk = tensor(k, dim, key_heads, tokens);
        auto * tv = tensor(v, dim, value_heads, tokens);
        auto * tg = tensor(g, 1, value_heads, tokens);
        auto * tb = tensor(beta, 1, value_heads, tokens);
        auto * ts = tensor(state, dim, dim, value_heads);
        auto * result = ggml_gated_delta_net(ctx.get(), tq, tk, tv, tg, tb, ts, 1);
        std::vector<float> packed(output_count + state_count);
        compute(ctx.get(), result, packed.data(), threads);
        std::memcpy(output, packed.data(), output_count * sizeof(float));
        std::memcpy(new_state, packed.data() + output_count, state_count * sizeof(float));
    });
}

} // extern C
