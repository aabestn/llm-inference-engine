#ifndef CACHE_MANAGER_H
#define CACHE_MANAGER_H

#include <vector>
#include <cstdint>

/**
 * @brief Host-side physical block manager interface for Key-Value Cache allocation.
 */
class CacheManager {
public:
    CacheManager(size_t num_blocks, size_t block_size);
    ~CacheManager() = default;

    // Allocate physical block IDs for a sequence
    std::vector<int32_t> allocate(size_t num_required_blocks);
    
    // Free allocated physical blocks back to pool
    void free(const std::vector<int32_t>& block_ids);
    
    // Get total remaining unallocated physical blocks
    size_t get_free_blocks() const;

private:
    size_t total_blocks_;
    size_t block_size_;
    std::vector<int32_t> free_pool_;
};

#endif // CACHE_MANAGER_H