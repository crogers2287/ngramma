#pragma once
// Local inference extension. Original tensors and model files remain read-only.
#include "json.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace flash_memory {
using json = nlohmann::json;
inline std::string hex(const unsigned char * p, size_t n) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    for (size_t i = 0; i < n; ++i) { result += digits[p[i] >> 4]; result += digits[p[i] & 15]; }
    return result;
}
inline std::string sha_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("FML cannot open model shard: " + path);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) throw std::runtime_error("FML SHA initialization failed");
    std::vector<char> buf(8 << 20);
    while (f) { f.read(buf.data(), buf.size()); if (EVP_DigestUpdate(ctx.get(), buf.data(), f.gcount()) != 1) throw std::runtime_error("FML SHA failed"); }
    if (!f.eof()) throw std::runtime_error("FML model read failed");
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int n = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest, &n) != 1) throw std::runtime_error("FML SHA failed");
    return hex(digest, n);
}
struct overlay {
    json header;
    uint32_t dim = 0;
    std::vector<int32_t> rows;
    std::vector<float> values;
    static std::unique_ptr<overlay> read() {
        const char * path = getenv("FLASH_MEMORY_OVERLAY");
        if (!path || !*path) return nullptr;
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("FML overlay missing");
        char magic[8]; uint32_t size = 0;
        f.read(magic, 8); f.read(reinterpret_cast<char *>(&size), 4);
        if (!f || memcmp(magic, "FMLROW1\0", 8) != 0 || size > (4u << 20)) throw std::runtime_error("FML invalid header");
        std::string text(size, '\0'); f.read(text.data(), size);
        auto result = std::make_unique<overlay>();
        result->header = json::parse(text);
        auto & h = result->header;
        if (h.at("schema") != "flash-memory-overlay/v1") throw std::runtime_error("FML unsupported format");
        const int64_t count = h.at("row_count").get<int64_t>();
        result->dim = h.at("row_dim").get<uint32_t>();
        if (count < 0 || count > 131072 || result->dim != 160) throw std::runtime_error("FML invalid geometry");
        const size_t payload_size = size_t(count) * (4 + result->dim * 4);
        std::vector<unsigned char> payload(payload_size);
        f.read(reinterpret_cast<char *>(payload.data()), payload.size());
        if (!f || f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("FML truncated or trailing payload");
        unsigned char checksum[EVP_MAX_MD_SIZE]; unsigned int n = 0;
        if (EVP_Digest(payload.data(), payload.size(), checksum, &n, EVP_sha256(), nullptr) != 1 || hex(checksum, n) != h.at("payload_sha256")) throw std::runtime_error("FML payload checksum mismatch");
        result->rows.resize(count); result->values.resize(count * result->dim);
        if (count) {
            memcpy(result->rows.data(), payload.data(), count * 4);
            memcpy(result->values.data(), payload.data() + count * 4, count * result->dim * 4);
        }
        for (size_t i = 0; i < result->rows.size(); ++i) {
            if (result->rows[i] < 0 || (i && result->rows[i] <= result->rows[i-1])) throw std::runtime_error("FML duplicate, unsorted, or negative row");
        }
        for (float v : result->values) if (!std::isfinite(v)) throw std::runtime_error("FML nonfinite value");
        return result;
    }
    void verify_model(const std::vector<std::string> & paths) const {
        const auto & shards = header.at("model_identity").at("shards");
        if (paths.size() != shards.size()) throw std::runtime_error("FML model shard count mismatch");
        for (size_t i = 0; i < paths.size(); ++i) {
            const auto & expected = shards.at(i);
            if (std::filesystem::canonical(paths[i]) != std::filesystem::canonical(expected.at("path").get<std::string>()) ||
                std::filesystem::file_size(paths[i]) != expected.at("bytes").get<uint64_t>() || sha_file(paths[i]) != expected.at("sha256")) {
                throw std::runtime_error("FML incompatible model shard: " + paths[i]);
            }
        }
    }
    void verify_table(const std::string & path, uint64_t offs, int type, int64_t dim_actual, int64_t nrows) const {
        const auto & tables = header.at("model_identity").at("table");
        if (tables.size() != 1) throw std::runtime_error("FML engine requires joined layout");
        const auto & t = tables.at(0);
        if (dim_actual != dim || t.at("offset") != offs || t.at("type") != type || t.at("shape").at(0) != dim || t.at("shape").at(1) != nrows ||
            std::filesystem::canonical(path) != std::filesystem::canonical(t.at("path").get<std::string>())) throw std::runtime_error("FML table mismatch");
        const auto & md = header.at("model_identity").at("architecture_metadata");
        const auto offsets = md.at("qwen4exp.ple.head_offsets").get<std::vector<int64_t>>();
        const auto sizes = md.at("qwen4exp.ple.head_vocab_sizes").get<std::vector<int64_t>>();
        if (offsets.size() != sizes.size() || offsets.empty()) throw std::runtime_error("FML head metadata invalid");
        for (int32_t row : rows) {
            auto it = std::upper_bound(offsets.begin(), offsets.end(), row);
            if (it == offsets.begin()) throw std::runtime_error("FML row outside head range");
            const size_t h = size_t(it - offsets.begin() - 1);
            if (row >= nrows || row >= offsets[h] + sizes[h]) throw std::runtime_error("FML row outside table or in padding");
        }
    }
    const float * find(int32_t row) const {
        auto it = std::lower_bound(rows.begin(), rows.end(), row);
        return it != rows.end() && *it == row ? values.data() + size_t(it - rows.begin()) * dim : nullptr;
    }
};
}
