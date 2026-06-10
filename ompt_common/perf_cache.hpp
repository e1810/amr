#ifndef AMR_OMPT_COMMON_PERF_CACHE_HPP
#define AMR_OMPT_COMMON_PERF_CACHE_HPP

#include <cstdint>

namespace perf_cache {

struct Delta {
    bool valid = false;
    std::uint64_t references = 0;
    std::uint64_t misses = 0;
    std::uint64_t hits = 0;
};

void configure(bool enabled);
bool start();
Delta stop();

}  // namespace perf_cache

#endif
