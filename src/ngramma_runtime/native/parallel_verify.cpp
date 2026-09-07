// Optional Linux/GCC ABI extension for the pinned archived overlay loader.
// It changes only scheduling of independent full-file SHA256 checks.
#include "memory-overlay.h"
#include <chrono>
#include <future>
#include <iostream>

namespace ngramma {
void verify_parallel(const flash_memory::overlay * self, const std::vector<std::string> & paths) {
    const auto started = std::chrono::steady_clock::now();
    const auto & shards = self->header.at("model_identity").at("shards");
    if (paths.size() != shards.size()) throw std::runtime_error("FML model shard count mismatch");
    uint64_t bytes = 0;
    // Preserve the original canonical-path and size checks before hashing.
    for (size_t i = 0; i < paths.size(); ++i) {
        const auto & expected = shards.at(i);
        if (std::filesystem::canonical(paths[i]) != std::filesystem::canonical(expected.at("path").get<std::string>()) ||
            std::filesystem::file_size(paths[i]) != expected.at("bytes").get<uint64_t>())
            throw std::runtime_error("FML incompatible model shard: " + paths[i]);
        bytes += expected.at("bytes").get<uint64_t>();
    }
    // At most three workers; every invocation reads every byte anew. No cache.
    for (size_t begin = 0; begin < paths.size(); begin += 3) {
        const size_t end = std::min(paths.size(), begin + 3);
        std::vector<std::future<std::string>> results;
        for (size_t i = begin; i < end; ++i)
            results.push_back(std::async(std::launch::async, [path = paths[i]] { return flash_memory::sha_file(path); }));
        for (size_t i = begin; i < end; ++i)
            if (results[i-begin].get() != shards.at(i).at("sha256"))
                throw std::runtime_error("FML incompatible model shard: " + paths[i]);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cerr << "NGRAMMA_MODEL_VERIFIED " << flash_memory::json({{"shards", paths.size()},
        {"workers", std::min<size_t>(3, paths.size())}, {"bytes", bytes}, {"seconds", seconds},
        {"full_file_sha256", true}, {"cache_reused", false}}).dump() << std::endl;
}
}

// The pinned library calls this weak member symbol through its PLT. Preserve
// its exact const-this/reference ABI; do not use with an unqualified runtime.
extern "C" void ngramma_overlay_verify_model(const flash_memory::overlay *, const std::vector<std::string> &)
    asm("_ZNK12flash_memory7overlay12verify_modelERKSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS7_EE");
extern "C" void ngramma_overlay_verify_model(const flash_memory::overlay * self, const std::vector<std::string> & paths) {
    ngramma::verify_parallel(self, paths);
}
