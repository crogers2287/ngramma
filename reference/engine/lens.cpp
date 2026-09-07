#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
using json = nlohmann::json;
struct capture_state { std::filesystem::path dir; std::vector<std::string> prefixes; json tensors = json::array(); int serial = 0; };
static bool capture(ggml_tensor * t, bool ask, void * opaque) {
    auto & state = *static_cast<capture_state *>(opaque);
    bool wanted = false;
    for (const auto & p : state.prefixes) if (std::string(t->name).rfind(p, 0) == 0) wanted = true;
    if (ask) return wanted;
    if (!wanted) return true;
    const std::string file = "tensor-" + std::to_string(state.serial++) + ".bin";
    std::vector<char> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());
    std::ofstream out(state.dir / file, std::ios::binary); out.write(data.data(), data.size());
    if (!out) throw std::runtime_error("Lens tensor write failed");
    state.tensors.push_back({{"name", t->name}, {"type", int(t->type)}, {"shape", {t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}, {"strides", {t->nb[0],t->nb[1],t->nb[2],t->nb[3]}}, {"file", file}});
    return true;
}
int main(int argc, char ** argv) {
    try {
        common_init(); common_params params;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
        params.warmup = false;
        capture_state state;
        params.cb_eval = capture; params.cb_eval_user_data = &state;
        llama_backend_init(); llama_numa_init(params.numa);
        auto init = common_init_from_params(params);
        auto * model = init->model(); auto * ctx = init->context();
        if (!model || !ctx) throw std::runtime_error("Lens model initialization failed");
        const auto * vocab = llama_model_get_vocab(model);
        const int nv = llama_vocab_n_tokens(vocab);
        std::cout << json({{"ready", true}, {"vocab_size", nv}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto job = json::parse(line);
                std::vector<llama_token> tokens = job.contains("tokens") ? job.at("tokens").get<std::vector<llama_token>>() : common_tokenize(ctx, job.at("text").get<std::string>(), false, true);
                for (auto token : tokens) if (token < 0 || token >= nv) throw std::runtime_error("Invalid token ID");
                if (job.value("tokenize_only", false)) { std::cout << json({{"tokens",tokens}}).dump() << std::endl; continue; }
                if (tokens.empty() || tokens.size() > llama_n_ctx(ctx)) throw std::runtime_error("Invalid sequence length");
                state.dir = job.at("output_dir").get<std::string>();
                std::filesystem::create_directories(state.dir);
                state.prefixes = job.value("capture", std::vector<std::string>{}); state.tensors = json::array(); state.serial = 0;
                llama_memory_clear(llama_get_memory(ctx), true);
                std::ofstream logits(state.dir / "logits.f32", std::ios::binary);
                int chunk_size = job.value("chunk_size", int(llama_n_batch(ctx)));
                if (chunk_size < 1 || chunk_size > int(llama_n_batch(ctx))) throw std::runtime_error("Invalid chunk size");
                for (size_t offset = 0; offset < tokens.size(); offset += chunk_size) {
                    const int count = std::min<size_t>(chunk_size, tokens.size() - offset);
                    llama_batch batch = llama_batch_init(count, 0, 1);
                    for (int i=0; i<count; ++i) common_batch_add(batch, tokens[offset+i], offset+i, {0}, true);
                    const int rc = llama_decode(ctx, batch);
                    llama_batch_free(batch);
                    if (rc) throw std::runtime_error("Lens decode failed: " + std::to_string(rc));
                    for (int i=0; i<count; ++i) {
                        const float * values = llama_get_logits_ith(ctx, i);
                        if (!values) throw std::runtime_error("Missing logits");
                        logits.write(reinterpret_cast<const char *>(values), nv * sizeof(float));
                    }
                }
                if (!logits) throw std::runtime_error("Lens logits write failed");
                std::ofstream meta(state.dir / "tensors.json"); meta << state.tensors.dump(2);
                std::ofstream token_file(state.dir / "tokens.json"); token_file << json(tokens).dump();
                std::cout << json({{"ok",true},{"tokens",tokens},{"vocab_size",nv},{"output_dir",state.dir.string()},{"tensors",state.tensors}}).dump() << std::endl;
            } catch (const std::exception & e) { std::cout << json({{"ok",false},{"error",e.what()}}).dump() << std::endl; }
        }
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << std::endl; return 1; }
}
