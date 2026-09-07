// Optional greedy generation extends the archived capture lens. No sampler,
// grammar, logit bias, answer forcing, or model-provided command execution.
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace generation_protocol {
using json = nlohmann::json;
constexpr int MAX_NEW_TOKENS = 1024;
constexpr int MAX_CONTEXT_TOKENS = 32768;
struct options {
    int max_new_tokens;
    int context_tokens;
    int top_k;
    bool compact;
    std::vector<int32_t> score_tokens;
};
static int bounded_integer(const json & value, int low, int high, const char * name) {
    if (!value.is_number_integer()) throw std::runtime_error(std::string(name) + " must be an integer");
    if (value.is_number_unsigned()) {
        const auto n = value.get<uint64_t>();
        if (n > uint64_t(high)) throw std::runtime_error(std::string(name) + " exceeds limit");
    }
    const auto n = value.get<int64_t>();
    if (n < low || n > high) throw std::runtime_error(std::string(name) + " outside bounds");
    return int(n);
}
static options parse(const json & job, int context, int vocabulary, int prompt_tokens) {
    if (!job.is_object() || !job.contains("generate") || !job["generate"].is_object())
        throw std::runtime_error("generate must be an object");
    if (context <= 0 || vocabulary <= 0) throw std::runtime_error("Invalid loaded model limits");
    if (job.value("tokenize_only", false)) throw std::runtime_error("generate and tokenize_only are mutually exclusive");
    const auto & g = job["generate"];
    const std::set<std::string> allowed = {"max_new_tokens", "context_tokens", "top_k", "score_tokens", "compact"};
    for (const auto & item : g.items()) if (!allowed.count(item.key()))
        throw std::runtime_error("Unknown generation option: " + item.key());
    options result;
    result.max_new_tokens = bounded_integer(g.at("max_new_tokens"), 1, MAX_NEW_TOKENS, "max_new_tokens");
    result.context_tokens = bounded_integer(g.value("context_tokens", json(std::min(context, MAX_CONTEXT_TOKENS))),
        1, std::min(context, MAX_CONTEXT_TOKENS), "context_tokens");
    if (prompt_tokens < 1 || prompt_tokens >= result.context_tokens)
        throw std::runtime_error("Generation requires a nonempty prompt and room for at least one new token");
    result.top_k = bounded_integer(g.value("top_k", json(std::min(5, vocabulary))), 0, std::min(100, vocabulary), "top_k");
    if (g.contains("compact") && !g["compact"].is_boolean()) throw std::runtime_error("compact must be boolean");
    result.compact = g.value("compact", true);
    if (g.contains("score_tokens")) {
        if (!g["score_tokens"].is_array() || g["score_tokens"].size() > 256)
            throw std::runtime_error("score_tokens must be an array of at most 256 IDs");
        std::set<int> seen;
        for (const auto & value : g["score_tokens"]) {
            const int token = bounded_integer(value, 0, vocabulary-1, "score token");
            if (!seen.insert(token).second) throw std::runtime_error("Duplicate requested score token");
            result.score_tokens.push_back(token);
        }
    }
    if (!result.compact && (!job.contains("output_dir") || !job["output_dir"].is_string() || job["output_dir"].get<std::string>().empty()))
        throw std::runtime_error("Noncompact generation requires output_dir");
    return result;
}
static std::vector<int32_t> token_ids(const json & value, int vocabulary) {
    if (!value.is_array() || value.size() > MAX_CONTEXT_TOKENS)
        throw std::runtime_error("tokens must be a bounded integer array");
    std::vector<int32_t> result;
    for (const auto & token : value) result.push_back(bounded_integer(token, 0, vocabulary-1, "prompt token"));
    return result;
}
static int greedy(const float * logits, int vocabulary) {
    if (!logits || vocabulary < 1) throw std::runtime_error("Missing generation logits");
    int best = 0;
    for (int i=0; i<vocabulary; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite generation logits");
        if (logits[i] > logits[best]) best = i; // ties resolve to lowest token ID
    }
    return best;
}
static json first_step(const float * logits, int vocabulary, const options & config) {
    const int best = greedy(logits, vocabulary);
    std::vector<int> indices(vocabulary);
    for (int i=0; i<vocabulary; ++i) indices[i]=i;
    std::partial_sort(indices.begin(), indices.begin()+config.top_k, indices.end(),
        [&](int a,int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    json top=json::array(), requested=json::array();
    for (int i=0; i<config.top_k; ++i) top.push_back({{"token_id",indices[i]},{"logit",logits[indices[i]]}});
    for (int token : config.score_tokens) requested.push_back({{"token_id",token},{"logit",logits[token]}});
    return {{"greedy_token_id",best},{"top_logits",top},{"requested_logits",requested}};
}
} // namespace generation_protocol

#ifndef NGRAMMA_LENS_PROTOCOL_ONLY
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <chrono>

// Derived from the archived research lens. Describe weight buffers without
// copying their bytes; tensor captures retain the original callback behavior.
static void describe_weights(ggml_tensor * t, std::set<ggml_tensor *> & seen, nlohmann::json & items) {
    if (!t || !seen.insert(t).second) return;
    if (std::string(t->name).find(".weight") != std::string::npos) {
        items.push_back({{"name", t->name}, {"type", int(t->type)},
            {"buffer", t->buffer ? ggml_backend_buffer_name(t->buffer) : "none"},
            {"shape", {t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}});
        return;
    }
    for (auto * source : t->src) describe_weights(source, seen, items);
}
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
    json weight_sources = json::array();
    std::set<ggml_tensor *> seen;
    describe_weights(t, seen, weight_sources);
    state.tensors.push_back({{"weight_sources", weight_sources}, {"name", t->name}, {"type", int(t->type)}, {"shape", {t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}, {"strides", {t->nb[0],t->nb[1],t->nb[2],t->nb[3]}}, {"file", file}});
    return true;
}

static json generate_job(const json & job, const std::vector<llama_token> & tokens,
                         llama_context * ctx, const llama_vocab * vocab, int nv,
                         capture_state & state) {
    const auto config = generation_protocol::parse(job, int(llama_n_ctx(ctx)), nv, int(tokens.size()));
    const int chunk = generation_protocol::bounded_integer(job.value("chunk_size", json(int(llama_n_batch(ctx)))),
                                                           1, int(llama_n_batch(ctx)), "chunk_size");
    state.prefixes.clear(); state.tensors=json::array(); state.serial=0; state.dir.clear();
    const bool write_files = job.contains("output_dir");
    if (write_files) {
        if (!job["output_dir"].is_string() || job["output_dir"].get<std::string>().empty())
            throw std::runtime_error("output_dir must be a nonempty string");
        state.dir=job["output_dir"].get<std::string>();
        std::filesystem::create_directories(state.dir);
    }
    if (!config.compact) state.prefixes=job.value("capture",std::vector<std::string>{});
    llama_memory_clear(llama_get_memory(ctx), true);
    const auto start=std::chrono::steady_clock::now();
    std::ofstream prefill, generation;
    if (!config.compact) {
        prefill.open(state.dir/"prefill-logits.f32",std::ios::binary);
        generation.open(state.dir/"generation-logits.f32",std::ios::binary);
        if (!prefill || !generation) throw std::runtime_error("Generation logits file open failed");
    }
    int prefill_calls=0, decode_calls=0;
    const float * next_logits=nullptr;
    for (size_t offset=0; offset<tokens.size(); offset+=chunk) {
        const int count=std::min<size_t>(chunk,tokens.size()-offset);
        llama_batch batch=llama_batch_init(count,0,1);
        for (int i=0;i<count;++i)
            common_batch_add(batch,tokens[offset+i],offset+i,{0},!config.compact || offset+i+1==tokens.size());
        const int rc=llama_decode(ctx,batch);
        llama_batch_free(batch);
        if (rc) throw std::runtime_error("Generation prefill failed: "+std::to_string(rc));
        ++prefill_calls;
        if (!config.compact) {
            for (int i=0;i<count;++i) {
                const float * values=llama_get_logits_ith(ctx,i);
                if (!values) throw std::runtime_error("Missing prefill logits");
                prefill.write(reinterpret_cast<const char *>(values),nv*sizeof(float));
            }
        }
        if (offset+count==tokens.size()) next_logits=llama_get_logits_ith(ctx,-1);
    }
    const auto first=generation_protocol::first_step(next_logits,nv,config);
    const int budget=std::min(config.max_new_tokens,config.context_tokens-int(tokens.size()));
    std::vector<llama_token> generated;
    std::string text;
    json eog=nullptr;
    std::string reason=budget<config.max_new_tokens ? "context_limit" : "max_new_tokens";
    for (int step=0;step<budget;++step) {
        const llama_token token=generation_protocol::greedy(next_logits,nv);
        if (!config.compact) generation.write(reinterpret_cast<const char *>(next_logits),nv*sizeof(float));
        generated.push_back(token);
        if (llama_vocab_is_eog(vocab,token)) { eog=token; reason="eog"; break; }
        text += common_token_to_piece(ctx,token,true);
        if (step+1==budget) break;
        llama_batch batch=llama_batch_init(1,0,1);
        common_batch_add(batch,token,tokens.size()+step,{0},true);
        const int rc=llama_decode(ctx,batch);
        llama_batch_free(batch);
        if (rc) throw std::runtime_error("Generation token decode failed: "+std::to_string(rc));
        ++decode_calls;
        next_logits=llama_get_logits_ith(ctx,-1);
    }
    if (!config.compact && (!prefill || !generation)) throw std::runtime_error("Generation logits write failed");
    std::string hex;
    constexpr char digits[]="0123456789abcdef";
    for (unsigned char byte : text) { hex+=digits[byte>>4]; hex+=digits[byte&15]; }
    json result={{"schema","ngramma.greedy-generation/v1"},{"ok",true},
        {"prompt_tokens",tokens},{"generated_tokens",generated},{"text",text},{"text_bytes_hex",hex},
        {"stop_reason",reason},{"eog_token",eog},{"stopped_on_eog",!eog.is_null()},
        {"generated_count",generated.size()},{"first_step",first},{"greedy",true},{"temperature",0},
        {"tie_break","lowest_token_id"},{"grammar",nullptr},{"fresh_state",true},{"model_reused",true},
        {"compact",config.compact},{"max_new_tokens",config.max_new_tokens},{"effective_new_token_budget",budget},
        {"context_tokens",config.context_tokens},{"chunk_size",chunk},{"vocab_size",nv},
        {"prefill_calls",prefill_calls},{"decode_calls",decode_calls},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
    if (write_files) {
        result["output_dir"]=state.dir.string();
        std::ofstream out(state.dir/"generation.json");
        out << result.dump(2,' ',false,json::error_handler_t::replace);
        if (!out) throw std::runtime_error("Generation metadata write failed");
        if (!config.compact) {
            std::ofstream meta(state.dir/"tensors.json"); meta << state.tensors.dump(2);
            std::ofstream prompt(state.dir/"tokens.json"); prompt << json(tokens).dump();
            if (!meta || !prompt) throw std::runtime_error("Generation capture metadata write failed");
        }
    }
    return result;
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
                std::vector<llama_token> tokens = job.contains("tokens") ? (job.contains("generate") ? generation_protocol::token_ids(job.at("tokens"),nv) : job.at("tokens").get<std::vector<llama_token>>()) : common_tokenize(ctx, job.at("text").get<std::string>(), false, true);
                for (auto token : tokens) if (token < 0 || token >= nv) throw std::runtime_error("Invalid token ID");
                if (job.contains("generate")) {
                    std::cout << generate_job(job,tokens,ctx,vocab,nv,state).dump(-1,' ',false,json::error_handler_t::replace) << std::endl;
                    continue;
                }
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

#endif // NGRAMMA_LENS_PROTOCOL_ONLY
