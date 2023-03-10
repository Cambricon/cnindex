/*************************************************************************
* Copyright (C) 2021 by Cambricon, Inc. All rights reserved
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* A part of this source code is referenced from google tensorflow project.
* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/bfc_allocator.cc
* Copyright 2015 The TensorFlow Authors. 
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*************************************************************************/

#include <atomic>
#include <cstring>

#include "bfc_allocator.h"

namespace cnindex {

constexpr BFCAllocator::ChunkHandle BFCAllocator::kInvalidChunkHandle;
constexpr uint64_t BFCAllocator::kMemDebugHistorySize;

static const int64_t kint64min = static_cast<int64_t>(~0x7FFFFFFFFFFFFFFFll);
std::string HumanReadableNumBytes(int64_t num_bytes) {
  if (num_bytes == kint64min) {
    // Special case for number with not representable negation.
    return "-8E";
  }

  const char* neg_str = (num_bytes < 0) ? "-" : "";
  if (num_bytes < 0) {
    num_bytes = -num_bytes;
  }

  // Special case for bytes.
  if (num_bytes < 1024) {
    // No fractions for bytes.
    char buf[8];  // Longest possible string is '-XXXXB'
    snprintf(buf, sizeof(buf), "%s%lldB", neg_str,
             static_cast<long long>(num_bytes));
    return std::string(buf);
  }

  static const char units[] = "KMGTPE";  // int64_t only goes up to E.
  const char* unit = units;
  while (num_bytes >= static_cast<int64_t>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
    CHECK(unit < units + strlen(units));
  }

  // We use SI prefixes.
  char buf[16];
  snprintf(buf, sizeof(buf), ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
           neg_str, num_bytes / 1024.0, *unit);
  return std::string(buf);
}

BFCAllocator::BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
                           bool allow_growth, const std::string& name,
                           bool garbage_collection)
    : garbage_collection_(garbage_collection),
      coalesce_regions_(sub_allocator->SupportsCoalescing()),
      sub_allocator_(sub_allocator),
      name_(name),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1) {
  if (allow_growth) {
    // 2MiB smallest initial allocation, unless total memory available
    // is less.
    curr_region_allocation_bytes_ =
        RoundedBytes(std::min(total_memory, size_t{2 << 20}));
  } else {
    curr_region_allocation_bytes_ = RoundedBytes(total_memory);
  }

  // Allocate the requested amount of memory.
  memory_limit_ = total_memory;
  stats_.bytes_limit = static_cast<int64_t>(total_memory);

  // Create a bunch of bins of various good sizes.

  // We create bins to fit all possible ranges that cover the
  // memory_limit_ starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    LOGD(BFCAllocator) << "Creating bin of max chunk size "
            << HumanReadableNumBytes(bin_size);
    new (BinFromIndex(b)) Bin(this, bin_size);
    CHECK_EQ(BinForSize(bin_size), BinFromIndex(b));
    CHECK_EQ(BinForSize(bin_size + 255), BinFromIndex(b));
    CHECK_EQ(BinForSize(bin_size * 2 - 1), BinFromIndex(b));
    if (b + 1 < kNumBins) {
      CHECK_NE(BinForSize(bin_size * 2), BinFromIndex(b));
    }
  }
}

BFCAllocator::~BFCAllocator() {
  // Return memory back.
  LOGD(BFCAllocator) << "Number of regions allocated: "
          << region_manager_.regions().size();
  for (const auto& region : region_manager_.regions()) {
    sub_allocator_->Free(region.ptr(), region.memory_size());
  }

  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b)->~Bin();
  }
}

BFCAllocator::Chunk* BFCAllocator::ChunkFromHandle(ChunkHandle h) {
  DCHECK_GE(h, 0);
  DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

const BFCAllocator::Chunk* BFCAllocator::ChunkFromHandle(ChunkHandle h) const {
  DCHECK_GE(h, 0);
  DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

bool BFCAllocator::Extend(size_t alignment, size_t rounded_bytes) {
  size_t available_bytes = memory_limit_ - total_region_allocated_bytes_;
  // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  // Do we have enough space to handle the client's request?
  // If not, fail immediately.
  if (rounded_bytes > available_bytes) {
    return false;
  }

  // If curr_region_allocation_bytes_ is not enough to satisfy the
  // allocation, keep multiplying by a power of two until that is
  // sufficient.
  bool increased_allocation = false;
  while (rounded_bytes > curr_region_allocation_bytes_) {
    curr_region_allocation_bytes_ *= 2;
    increased_allocation = true;
  }

  // Try allocating.
  size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
  size_t bytes_received;
  void* mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
  if (mem_addr == nullptr && !started_backpedal_) {
    // Only backpedal once.
    started_backpedal_ = true;

    static constexpr float kBackpedalFactor = 0.9;

    // Try allocating less memory.
    while (mem_addr == nullptr) {
      bytes = RoundedBytes(bytes * kBackpedalFactor);
      if (bytes < rounded_bytes) break;
      mem_addr = sub_allocator_->Alloc(alignment, bytes, &bytes_received);
    }
  }

  if (mem_addr == nullptr) {
    return false;
  }

  if (!increased_allocation) {
    // Increase the region size of the next required allocation.
    curr_region_allocation_bytes_ *= 2;
  }

  LOGD(BFCAllocator) << "Extending allocation by "
          << HumanReadableNumBytes(bytes_received) << " bytes.";

  total_region_allocated_bytes_ += bytes_received;
  LOGD(BFCAllocator) << "Total allocated bytes: "
          << HumanReadableNumBytes(total_region_allocated_bytes_);

  LOGD(BFCAllocator) << "Allocated memory at " << mem_addr << " to "
          << static_cast<void*>(static_cast<char*>(mem_addr) + bytes_received);

  AllocationRegion* maybe_extended_region = nullptr;
  if (coalesce_regions_) {
    maybe_extended_region =
        region_manager_.AddOrExtendAllocationRegion(mem_addr, bytes_received);
  } else {
    region_manager_.AddAllocationRegion(mem_addr, bytes_received);
  }

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  BFCAllocator::Chunk* c = ChunkFromHandle(h);
  c->ptr = mem_addr;
  c->size = bytes_received;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;
  c->freed_at_count = 0;

  region_manager_.set_handle(c->ptr, h);

  // If the region was extended, then there exists a previous chunk that should
  // be linked to the new chunk.
  if (maybe_extended_region != nullptr) {
    ChunkHandle prev =
        maybe_extended_region->get_handle(maybe_extended_region->ptr());
    BFCAllocator::Chunk* prev_chunk = ChunkFromHandle(prev);
    // Find the last recorded chunk in the extended region.
    while (prev_chunk->next != kInvalidChunkHandle) {
      prev = prev_chunk->next;
      prev_chunk = ChunkFromHandle(prev);
    }
    c->prev = prev;
    prev_chunk->next = h;
  }

  // Maybe merge adjacent chunks and insert the chunk into the right bin.
  InsertFreeChunkIntoBin(TryToCoalesce(h, /*ignore_freed_at=*/false));

  return true;
}

BFCAllocator::ChunkHandle BFCAllocator::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    return h;
  } else {
    ChunkHandle h = chunks_.size();
    chunks_.resize(h + 1);
    return h;
  }
}

void BFCAllocator::DeallocateChunk(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  c->allocation_id = -1;
  c->bin_num = kInvalidBinNum;
  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

void* BFCAllocator::AllocateRawInternalWithRetry(
    size_t unused_alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  // Fast path: Try once to allocate without getting the retry_helper_ involved
  uint64_t freed_by_count = 0;
  if (allocation_attr.freed_by_func != nullptr) {
    freed_by_count = (*allocation_attr.freed_by_func)();
  }
  void* r =
      AllocateRawInternal(unused_alignment, num_bytes, false, freed_by_count);
  if (r != nullptr) {
    return r;
  } else {
    static const int64_t kMaxMillisToWait = 10000;  // 10 seconds
    r = retry_helper_.AllocateRaw(
        [this, &allocation_attr](size_t a, size_t nb, bool v) {
          uint64_t freed_by_count = 0;
          if (allocation_attr.freed_by_func != nullptr) {
            freed_by_count = (*allocation_attr.freed_by_func)();
          }
          return AllocateRawInternal(a, nb, v, freed_by_count);
        },
        kMaxMillisToWait, unused_alignment, num_bytes);
    return r;
  }
}

void* BFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes,
                                const AllocationAttributes& allocation_attr) {
  LOGD(BFCAllocator) << "AllocateRaw " << Name() << "  " << num_bytes;
  if (!allocation_attr.retry_on_failure) {
    // Return immediately upon the first failure if this is for allocating an
    // optional scratch space.
    bool dump_log_on_failure = true;  // VLOG_IS_ON(2);
    uint64_t freed_by_count = 0;
    if (allocation_attr.freed_by_func != nullptr) {
      freed_by_count = (*allocation_attr.freed_by_func)();
    }
    void* result = AllocateRawInternal(unused_alignment, num_bytes,
                                       dump_log_on_failure, freed_by_count);
    if (result == nullptr) {
      static std::atomic<int32_t> log_counter{0};
      int32_t counter_value = log_counter.load(std::memory_order_relaxed);
      if (counter_value < 10) {
        log_counter.store(counter_value + 1, std::memory_order_relaxed);
        LOGW(BFCAllocator)
            << "Allocator (" << Name() << ") ran out of memory trying "
            << "to allocate " << HumanReadableNumBytes(num_bytes)
            << " with freed_by_count=" << freed_by_count

            << ". The caller indicates that this is not a failure, but"
            << " may mean that there could be performance gains if more"
            << " memory were available.";
      }
    }
    return result;
  } else {
    return AllocateRawInternalWithRetry(unused_alignment, num_bytes,
                                        allocation_attr);
  }
}

// static
size_t BFCAllocator::RoundedBytes(size_t bytes) {
  size_t rounded_bytes =
      (kMinAllocationSize *
       ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
  DCHECK_EQ(size_t{0}, rounded_bytes % kMinAllocationSize);
  return rounded_bytes;
}

bool BFCAllocator::DeallocateFreeRegions(size_t rounded_bytes)
    EXCLUSIVE_LOCKS_REQUIRED(lock_) {
  // Do nothing if garbage collection is off.
  if (!garbage_collection_) {
    return false;
  }

  // Searching for free regions.
  std::unordered_set<void*> free_region_ptrs;
  size_t total_free_bytes = 0;
  for (const AllocationRegion& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    bool any_use = false;
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        any_use = true;
        break;
      }
      h = c->next;
    }

    if (!any_use) {
      LOGD(BFCAllocator) << "Found free region with ptr = " << region.ptr();
      free_region_ptrs.insert(region.ptr());
      total_free_bytes += region.memory_size();
    }
  }

  if (total_free_bytes == 0) {
    return false;
  }

  // Rough estimation to check whether deallocation can help.
  size_t available_bytes =
      memory_limit_ - total_region_allocated_bytes_ + total_free_bytes;
  if (rounded_bytes > available_bytes) {
    return false;
  }

  LOGW(BFCAllocator) << "Garbage collection: deallocate free memory regions"
               << " (i.e., allocations) so that we can re-allocate a larger"
               << " region to avoid OOM due to memory fragmentation. If you"
               << " see this message frequently, you are running near the"
               << " threshold of the available device memory and re-allocation"
               << " may incur great performance overhead. You may try smaller"
               << " batch sizes to observe the performance impact."
               << " Set TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to"
               << " disable this feature.";

  // Deallocate free regions.
  DeallocateRegions(free_region_ptrs);

  return true;
}

void BFCAllocator::DeallocateRegions(
    const std::unordered_set<void*>& region_ptrs)
    EXCLUSIVE_LOCKS_REQUIRED(lock_) {
  // Explicitly remove the const qualifier as some compilers disallow passing
  // const_iterator to std::vector::erase(), which is used in
  // RemoveAllocationRegion().
  auto regions =
      const_cast<std::vector<AllocationRegion>*>(&region_manager_.regions());
  auto it = regions->begin();
  while (it != regions->end()) {
    if (region_ptrs.count(it->ptr()) == 0) {
      ++it;
      continue;
    }

    LOGD(BFCAllocator) << "Deallocate region with ptr = " << it->ptr();
    // Remove all chunk registrations from Bins.
    ChunkHandle h = region_manager_.get_handle(it->ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->bin_num != kInvalidBinNum) {
        RemoveFreeChunkFromBin(h);
      }
      auto h_to_delete = h;
      h = c->next;
      DeleteChunk(h_to_delete);
    }

    // Deallocate the memory.
    sub_allocator_->Free(it->ptr(), it->memory_size());
    total_region_allocated_bytes_ -= it->memory_size();
    it = region_manager_.RemoveAllocationRegion(it);
  }
}

void* BFCAllocator::AllocateRawInternal(size_t unused_alignment,
                                        size_t num_bytes,
                                        bool dump_log_on_failure,
                                        uint64_t freed_before) {
  if (num_bytes == 0) {
    LOGD(BFCAllocator) << "tried to allocate 0 bytes";
    return nullptr;
  }
  // First, always allocate memory of at least kMinAllocationSize
  // bytes, and always allocate multiples of kMinAllocationSize bytes
  // so all memory addresses are nicely byte aligned.
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  std::lock_guard<std::mutex> l(lock_);
  if (!timestamped_chunks_.empty()) {
    // Merge timestamped chunks whose counts have become safe for general use.
    MergeTimestampedChunks(0);
  }
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
  if (ptr != nullptr) {
    // AddTraceMe("MemoryAllocation", ptr);
    return ptr;
  }

  // Try to extend
  if (Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
    if (ptr != nullptr) {
      // AddTraceMe("MemoryAllocation", ptr);
      return ptr;
    }
  }

  if ((freed_before == 0) && (!timestamped_chunks_.empty())) {
    // We're unable to satisfy an allocation request without a specific
    // timestamp requirement.  Rather than fail, try merging any held-out
    // timestamped chunks more aggressively until a free chunk of the necessary
    // size is formed.
    if (MergeTimestampedChunks(rounded_bytes)) {
      ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
      if (ptr != nullptr) {
        // AddTraceMe("MemoryAllocation", ptr);
        return ptr;
      }
    }
  }

  // Reaching this point means that no chunks can satisfy the request. Also,
  // the unallocated bytes cannot satisfy the request. Before giving up, let's
  // try deallocating free regions so that suballocator can combine them with
  // the unallocated bytes and form a larger region.
  if (DeallocateFreeRegions(rounded_bytes) &&
      Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
    if (ptr != nullptr) {
      // AddTraceMe("MemoryAllocation", ptr);
      return ptr;
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  // MaybeWriteMemoryMap();
  if (dump_log_on_failure) {
    LOGW(BFCAllocator)
        << "Allocator (" << Name() << ") ran out of memory trying "
        << "to allocate " << HumanReadableNumBytes(num_bytes)
        << " (rounded to " << rounded_bytes << ")"
        << "requested by op "
        << ScopedMemoryDebugAnnotation::CurrentAnnotation().pending_op_name
        << "\nIf the cause is memory fragmentation maybe the environment "
        << "variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will "
        << "improve the situation. \nCurrent allocation summary follows."
        << "\nCurrent allocation summary follows.";
    // DumpMemoryLog(rounded_bytes);
    LOGW(BFCAllocator) << RenderOccupancy();
  }
  return nullptr;
}

int64_t BFCAllocator::LargestFreeChunk() {
  for (int i = kNumBins - 1; i >= 0; i--) {
    if (!BinFromIndex(i)->free_chunks.empty()) {
      return ChunkFromHandle(*BinFromIndex(i)->free_chunks.rbegin())->size;
    }
  }
  return 0;
}

double BFCAllocator::GetFragmentation() {
  int64_t bytes_available = total_region_allocated_bytes_ - stats_.bytes_in_use;
  DCHECK_GT(bytes_available, 0);
  return static_cast<double>(bytes_available - LargestFreeChunk()) /
         bytes_available;
}
/*
void BFCAllocator::AddTraceMe(absl::string_view traceme_name, const void* ptr) {
  BFCAllocator::Chunk* chunk = ChunkFromHandle(region_manager_.get_handle(ptr));
  AddTraceMe(traceme_name, chunk->ptr, chunk->requested_size, chunk->size);
}

void BFCAllocator::AddTraceMe(absl::string_view traceme_name,
                              const void* chunk_ptr, int64_t req_bytes,
                              int64_t alloc_bytes) {
  cnindex::profiler::TraceMe::InstantActivity(
      [this, traceme_name, chunk_ptr, req_bytes, alloc_bytes]()
          TF_NO_THREAD_SAFETY_ANALYSIS {
            int64_t bytes_available =
                memory_limit_ - stats_.bytes_reserved - stats_.bytes_in_use;
            const auto& annotation =
                ScopedMemoryDebugAnnotation::CurrentAnnotation();
            std::string tensor_shape;
            if (annotation.pending_shape) {
              tensor_shape = annotation.pending_shape->DebugString();
            }
            return cnindex::profiler::TraceMeEncode(
                traceme_name, {{"allocator_name", name_},
                               {"bytes_reserved", stats_.bytes_reserved},
                               {"bytes_allocated", stats_.bytes_in_use},
                               {"bytes_available", bytes_available},
                               {"fragmentation", GetFragmentation()},
                               {"peak_bytes_in_use", stats_.peak_bytes_in_use},
                               {"requested_bytes", req_bytes},
                               {"allocation_bytes", alloc_bytes},
                               {"addr", reinterpret_cast<uint64_t>(chunk_ptr)},
                               {"tf_op", annotation.pending_op_name},
                               {"id", annotation.pending_step_id},
                               {"region_type", annotation.pending_region_type},
                               {"data_type", annotation.pending_data_type},
                               {"shape", tensor_shape}});
          },
      /*level=profiler::TraceMeLevel::kInfo);
}
*/
void* BFCAllocator::FindChunkPtr(BinNum bin_num, size_t rounded_bytes,
                                 size_t num_bytes, uint64_t freed_before) {
  // First identify the first bin that could satisfy rounded_bytes.
  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      const BFCAllocator::ChunkHandle h = (*citer);
      BFCAllocator::Chunk* chunk = ChunkFromHandle(h);
      DCHECK(!chunk->in_use());
      if (freed_before > 0 && freed_before < chunk->freed_at_count) {
        continue;
      }
      if (chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use, so remove
        // it from the free bin structure prior to using.
        RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably large
        // pieces, do so.  In any case don't waste more than
        // kMaxInternalFragmentation bytes on padding this alloc.
        const int64_t kMaxInternalFragmentation = 128 << 20;  // 128mb
        if (chunk->size >= rounded_bytes * 2 ||
            static_cast<int64_t>(chunk->size) - rounded_bytes >=
                kMaxInternalFragmentation) {
          SplitChunk(h, rounded_bytes);
          chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
        }

        // The requested size of the returned chunk is what the user
        // has allocated.
        chunk->requested_size = num_bytes;
        // Assign a unique id and increment the id counter, marking the
        // chunk as being in use.
        chunk->allocation_id = next_allocation_id_++;

        // Update stats.
        ++stats_.num_allocs;
        stats_.bytes_in_use += chunk->size;
        stats_.peak_bytes_in_use =
            std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
        stats_.largest_alloc_size =
            std::max<std::size_t>(stats_.largest_alloc_size, chunk->size);
        if (ShouldRecordOpName()) {
          const auto& annotation =
              ScopedMemoryDebugAnnotation::CurrentAnnotation();
          chunk->op_name = annotation.pending_op_name;
          if (!annotation.pending_op_name) {
            LOGD(BFCAllocator) << "missing pending_op_name for " << Name()
                    << " reading addr "
                    << static_cast<const void*>(&annotation.pending_op_name)
                    << "\n";
                    //<< CurrentStackTrace();
          }
          chunk->step_id = annotation.pending_step_id;
          chunk->action_count = ++action_counter_;
          uint64_t slot = chunk->action_count % kMemDebugHistorySize;
          size_history_[slot] = stats_.bytes_in_use;
        }

        // LOGI(BFCAllocator) << "Returning: " << chunk->ptr;
        // if (VLOG_IS_ON(4)) {
          // LOGI(BFCAllocator) << "A: " << RenderOccupancy();
        // }
        return chunk->ptr;
      }
    }
  }

  return nullptr;
}

void BFCAllocator::SplitChunk(BFCAllocator::ChunkHandle h, size_t num_bytes) {
  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));

  // Create a new chunk starting num_bytes after c
  BFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // It inherits the freed time.
  new_chunk->freed_at_count = c->freed_at_count;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  BFCAllocator::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk);
}

void BFCAllocator::DeallocateRaw(void* ptr) {
  LOGD(BFCAllocator) << "DeallocateRaw " << Name() << " "
          << (ptr ? RequestedSize(ptr) : 0);
  DeallocateRawInternal(ptr);
  retry_helper_.NotifyDealloc();
}

void BFCAllocator::DeallocateRawInternal(void* ptr) {
  if (ptr == nullptr) {
    LOGD(BFCAllocator) << "tried to deallocate nullptr";
    return;
  }
  std::lock_guard<std::mutex> l(lock_);

  // Find the chunk from the ptr.
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle);
  // Record chunk information before it's freed.
  Chunk* chunk = ChunkFromHandle(h);
  void* chunk_ptr = chunk->ptr;
  int64_t req_bytes = chunk->requested_size;
  int64_t alloc_bytes = chunk->size;

  MarkFree(h);

  // Consider coalescing it.
  if (/*timing_counter_*/0) {
    InsertFreeChunkIntoBin(h);
    timestamped_chunks_.push_back(h);
  } else {
    InsertFreeChunkIntoBin(TryToCoalesce(h, false));
  }

  // TraceMe needs to be added after MarkFree and InsertFreeChunkIntoBin for
  // correct aggregation stats (bytes_in_use, fragmentation).
  // AddTraceMe("MemoryDeallocation", chunk_ptr, req_bytes, alloc_bytes);

  // if (VLOG_IS_ON(4)) {
    // LOGI(BFCAllocator) << "F: " << RenderOccupancy();
  // }
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void BFCAllocator::Merge(BFCAllocator::ChunkHandle h1,
                         BFCAllocator::ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  // We can only merge chunks that are not in use.
  CHECK(!c1->in_use() && !c2->in_use());

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  BFCAllocator::ChunkHandle h3 = c2->next;
  c1->next = h3;
  CHECK(c2->prev == h1);
  if (h3 != kInvalidChunkHandle) {
    BFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

  // Pick latest free time.
  c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

  DeleteChunk(h2);
}

void BFCAllocator::DeleteChunk(ChunkHandle h) {
  // Delete h and cleanup all state
  Chunk* c = ChunkFromHandle(h);
  //  LOGI(BFCAllocator) << "Removing: " << c->ptr;
  region_manager_.erase(c->ptr);
  DeallocateChunk(h);
}

void BFCAllocator::InsertFreeChunkIntoBin(BFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));
  BinNum bin_num = BinNumForSize(c->size);
  Bin* new_bin = BinFromIndex(bin_num);
  c->bin_num = bin_num;
  new_bin->free_chunks.insert(h);
}

void BFCAllocator::RemoveFreeChunkIterFromBin(
    BFCAllocator::Bin::FreeChunkSet* free_chunks,
    const BFCAllocator::Bin::FreeChunkSet::iterator& citer) {
  ChunkHandle h = *citer;
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
  free_chunks->erase(citer);
  c->bin_num = kInvalidBinNum;
}

void BFCAllocator::RemoveFreeChunkFromBin(BFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
  CHECK_GT(BinFromIndex(c->bin_num)->free_chunks.erase(h), 0)
      << "Could not find chunk in bin";
  c->bin_num = kInvalidBinNum;
}

void BFCAllocator::MarkFree(BFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(c->in_use() && (c->bin_num == kInvalidBinNum));

  // Mark the chunk as no longer in use.
  c->allocation_id = -1;

  // Optionally record the free time.
  /*if (timing_counter_) {
    c->freed_at_count = timing_counter_->next();
  }*/

  // Updates the stats.
  stats_.bytes_in_use -= c->size;

  if (ShouldRecordOpName()) {
    c->action_count = ++action_counter_;
    uint64_t slot = c->action_count % kMemDebugHistorySize;
    size_history_[slot] = stats_.bytes_in_use;
  }
}

BFCAllocator::ChunkHandle BFCAllocator::TryToCoalesce(ChunkHandle h,
                                                      bool ignore_freed_at) {
  Chunk* c = ChunkFromHandle(h);
  if ((!ignore_freed_at) && c->freed_at_count > 0) return h;
  ChunkHandle coalesced_chunk = h;

  // If the next chunk is free, merge it into c and delete it.
  if (c->next != kInvalidChunkHandle && !ChunkFromHandle(c->next)->in_use()) {
    Chunk* n = ChunkFromHandle(c->next);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      LOGD(BFCAllocator) << "Merging c->next " << n->ptr << " with c " << c->ptr;
      RemoveFreeChunkFromBin(c->next);
      Merge(h, c->next);
    }
  }

  // If the previous chunk is free, merge c into it and delete c.
  if (c->prev != kInvalidChunkHandle && !ChunkFromHandle(c->prev)->in_use()) {
    Chunk* n = ChunkFromHandle(c->prev);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      LOGD(BFCAllocator) << "Merging c " << c->ptr << " into c->prev " << n->ptr;
      coalesced_chunk = c->prev;
      RemoveFreeChunkFromBin(c->prev);
      Merge(c->prev, h);
    }
  }

  return coalesced_chunk;
}

void BFCAllocator::SetSafeFrontier(uint64_t count) {
  uint64_t current = safe_frontier_.load(std::memory_order_relaxed);
  while (count > current) {
    if (safe_frontier_.compare_exchange_strong(current, count)) {
      retry_helper_.NotifyDealloc();
      return;
    } else {
      current = safe_frontier_.load(std::memory_order_relaxed);
    }
  }
}

bool BFCAllocator::MergeTimestampedChunks(size_t required_bytes) {
  LOGD(BFCAllocator) << "MergeTimestampedChunks queue_len=" << timestamped_chunks_.size()
          << " required_bytes=" << required_bytes;
  bool satisfied = (required_bytes == 0);
  std::vector<void*> to_merge;
  std::deque<ChunkHandle> new_ts_queue;
  while (!timestamped_chunks_.empty()) {
    ChunkHandle h = timestamped_chunks_.front();
    timestamped_chunks_.pop_front();
    DCHECK_NE(h, kInvalidChunkHandle);
    Chunk* c = ChunkFromHandle(h);
    // It's possible this chunk has already been merged so refetch and retest
    // the handle.
    h = region_manager_.get_handle(c->ptr);
    if (h == kInvalidChunkHandle) {
      continue;
    }
    if (c->in_use() || (c->bin_num == kInvalidBinNum)) {
      // This chunk has already been reallocated.
      continue;
    }
    if (c->freed_at_count == 0) {
      to_merge.push_back(c->ptr);
      continue;
    }
    // Chunk should be free and assigned to a bin.
    DCHECK_NE(c->bin_num, kInvalidBinNum);
    if (c->freed_at_count < safe_frontier_) {
      c->freed_at_count = 0;
      to_merge.push_back(c->ptr);
    } else if (required_bytes > 0) {
      to_merge.push_back(c->ptr);
    } else {
      new_ts_queue.push_back(h);
    }
  }
  DCHECK(timestamped_chunks_.empty());
  std::swap(timestamped_chunks_, new_ts_queue);

  // At this point all candidate chunks have been moved from timestamped_chunks_
  // to to_merge.  If this is a standard merge (required_bytes == 0) then
  // merge them all, otherwise merge just until a Chunk of the required size
  // is produced.
  for (int ci = 0, end = to_merge.size(); ci < end; ++ci) {
    void* ptr = to_merge[ci];
    // It's possible that the Chunk associated with this memory location got
    // merged and deallocated in a prior iteration so refetch the handle and
    // retest.
    ChunkHandle h = region_manager_.get_handle(ptr);
    if (h == kInvalidChunkHandle) continue;
    if (required_bytes == 0 || !satisfied) {
      Chunk* c = ChunkFromHandle(h);
      DCHECK_NE(c->bin_num, kInvalidBinNum);
      DCHECK(!c->in_use());
      RemoveFreeChunkFromBin(h);
      ChunkHandle new_h = TryToCoalesce(h, (required_bytes > 0));
      InsertFreeChunkIntoBin(new_h);
      if (required_bytes > 0) {
        c = ChunkFromHandle(new_h);
        if (new_h != h && c->freed_at_count > 0) {
          timestamped_chunks_.push_back(new_h);
        }
        if (c->size >= required_bytes) {
          satisfied = true;
        }
      }
    } else {
      // We were force merging Chunks with unsafe timestamps, but managed
      // to create a satisfying Chunk so just requeue the rest.
      timestamped_chunks_.push_back(h);
    }
  }
  return satisfied;
}

bool BFCAllocator::TracksAllocationSizes() const { return true; }

size_t BFCAllocator::RequestedSize(const void* ptr) const {
  CHECK(ptr);
  std::lock_guard<std::mutex> l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for requested size of pointer we never allocated: " << ptr;
  const BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t BFCAllocator::AllocatedSize(const void* ptr) const {
  std::lock_guard<std::mutex> l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocated size of pointer we never allocated: " << ptr;
  const BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->size;
}

int64_t BFCAllocator::AllocationId(const void* ptr) const {
  std::lock_guard<std::mutex> l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocation id of pointer we never allocated: " << ptr;
  const BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->allocation_id;
}

namespace {

void RenderRegion(char* rendered, const size_t resolution,
                  const size_t total_render_size, const size_t offset,
                  const void* base_ptr, const void* ptr, const size_t size,
                  const char c) {
  const char* base_ptr_c = static_cast<const char*>(base_ptr);
  const char* ptr_c = static_cast<const char*>(ptr);

  size_t start_location =
      ((ptr_c - base_ptr_c + offset) * resolution) / total_render_size;
  CHECK_GE(start_location, 0);
  CHECK_LT(start_location, resolution);
  size_t end_location =
      ((ptr_c + size - 1 - base_ptr_c + offset) * resolution) /
      total_render_size;
  CHECK_GE(end_location, 0);
  CHECK_LT(end_location, resolution);

  for (size_t i = start_location; i <= end_location; ++i) {
    rendered[i] = c;
  }
}

}  // namespace

std::string BFCAllocator::RenderOccupancy() {
  // Make a buffer for the ASCII-art representation.
  const size_t resolution = 100;
  char rendered[resolution];

  // Compute the total region size to render over
  size_t total_region_size = 0;
  for (const auto& region : region_manager_.regions()) {
    total_region_size += region.memory_size();
  }

  if (total_region_size == 0) {
    return "<allocator contains no memory>";
  }

  // Start out with everything empty
  RenderRegion(rendered, resolution, total_region_size, 0, nullptr, nullptr,
               total_region_size, '_');

  size_t region_offset = 0;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    // Then render each chunk left to right.
    while (h != kInvalidChunkHandle) {
      Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        // Render the wasted space
        size_t wasted = c->size - c->requested_size;
        if (wasted > 0) {
          RenderRegion(rendered, resolution, total_region_size,
                       region_offset + c->requested_size, region.ptr(), c->ptr,
                       wasted, 'x');
        }
        // Then the occupied space
        RenderRegion(rendered, resolution, total_region_size, region_offset,
                     region.ptr(), c->ptr, c->requested_size, '*');
      }
      h = c->next;
    }
    region_offset += region.memory_size();
  }

  return std::string(rendered, resolution);
}
/*
void BFCAllocator::DumpMemoryLog(size_t num_bytes) {
  const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
  LOGI(BFCAllocator) << "BFCAllocator dump for " << Name();
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    CHECK_EQ(b->free_chunks.size(),
             bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);

    LOGI(BFCAllocator) << "Bin (" << b->bin_size
              << "): \tTotal Chunks: " << bin_info.total_chunks_in_bin
              << ", Chunks in use: " << bin_info.total_chunks_in_use << ". "
              << HumanReadableNumBytes(bin_info.total_bytes_in_bin)
              << " allocated for chunks. "
              << HumanReadableNumBytes(bin_info.total_bytes_in_use)
              << " in use in bin. "
              << HumanReadableNumBytes(
                     bin_info.total_requested_bytes_in_use)
              << " client-requested in use in bin.";
  }

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  Bin* b = BinForSize(num_bytes);

  LOGI(BFCAllocator) << "Bin for " << HumanReadableNumBytes(num_bytes)
            << " was " << HumanReadableNumBytes(b->bin_size)
            << ", Chunk State: ";

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    LOGI(BFCAllocator) << c->DebugString(this, true);
  }

  // Next show the chunks that are in use, and also summarize their
  // number by size.
  std::map<size_t, int> in_use_by_size;
  for (const auto& region : region_manager_.regions()) {
    LOGI(BFCAllocator) << "Next region of size " << region.memory_size();
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        in_use_by_size[c->size]++;
      }
      string buf = strings::StrCat(
          (c->in_use() ? "InUse" : "Free "), " at ",
          strings::Hex(reinterpret_cast<uint64_t>(c->ptr)), " of size ", c->size);
      if (ShouldRecordOpName()) {
        strings::StrAppend(&buf, " by op ", c->GetDebugOpName(),
                           " action_count ", c->action_count, " step ",
                           c->step_id);
      }
      strings::StrAppend(&buf, " next ", c->next);
      if (timing_counter_) {
        strings::StrAppend(&buf, " freed_at_count ", c->freed_at_count);
      }
      LOGI(BFCAllocator) << buf;
      h = c->next;
    }
  }

  LOGI(BFCAllocator) << "     Summary of in-use Chunks by size: ";
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOGI(BFCAllocator) << it.second << " Chunks of size " << it.first << " totalling "
              << HumanReadableNumBytes(it.first * it.second);
    total_bytes += (it.first * it.second);
  }
  LOGI(BFCAllocator) << "Sum Total of in-use chunks: "
            << HumanReadableNumBytes(total_bytes);
  LOGI(BFCAllocator) << "total_region_allocated_bytes_: "
            << total_region_allocated_bytes_
            << " memory_limit_: " << memory_limit_ << " available bytes: "
            << (memory_limit_ - total_region_allocated_bytes_)
            << " curr_region_allocation_bytes_: "
            << curr_region_allocation_bytes_;
  LOGI(BFCAllocator) << "Stats: \n" << stats_.DebugString();
}

void BFCAllocator::MaybeWriteMemoryMap() {
  const char* gpu_memory_map_file = std::getenv("TF_BFC_MEMORY_DUMP");
  if (gpu_memory_map_file != nullptr) {
    std::unique_ptr<WritableFile> dump_file;
    string file_name = strings::StrCat(gpu_memory_map_file, "_", Name(), ".",
                                       Env::Default()->NowMicros());
    Status status = Env::Default()->NewWritableFile(file_name, &dump_file);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to open file " << file_name;
      return;
    }
    MemoryDump md = RecordMemoryMapInternal();
    status = dump_file->Append(md.SerializeAsString());
    if (!status.ok()) {
      LOG(ERROR) << "Error on writing to file " << gpu_memory_map_file << ": "
                 << status;
    }
  }
}

MemoryDump BFCAllocator::RecordMemoryMap() {
  std::lock_guard<std::mutex> l(lock_);
  return RecordMemoryMapInternal();
}

MemoryDump BFCAllocator::RecordMemoryMapInternal() {
  MemoryDump md;
  md.set_allocator_name(Name());

  // Record the general stats
  MemAllocatorStats* mas = md.mutable_stats();
  mas->set_num_allocs(stats_.num_allocs);
  mas->set_bytes_in_use(stats_.bytes_in_use);
  mas->set_peak_bytes_in_use(stats_.peak_bytes_in_use);
  mas->set_largest_alloc_size(stats_.largest_alloc_size);

  // Record summary data for every bin.
  const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    DCHECK_EQ(b->free_chunks.size(),
              bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);
    BinSummary* bs = md.add_bin_summary();
    bs->set_bin(bin_num);
    bs->set_total_bytes_in_use(bin_info.total_bytes_in_use);
    bs->set_total_bytes_in_bin(bin_info.total_bytes_in_bin);
    bs->set_total_chunks_in_use(bin_info.total_chunks_in_use);
    bs->set_total_chunks_in_bin(bin_info.total_chunks_in_bin);
  }

  // Record state of every defined Chunk.
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      MemChunk* mc = md.add_chunk();
      mc->set_in_use(c->in_use());
      mc->set_address(reinterpret_cast<uint64_t>(c->ptr));
      mc->set_size(c->size);
      mc->set_requested_size(c->requested_size);
      mc->set_bin(c->bin_num);
      mc->set_op_name(c->GetDebugOpName());
      mc->set_step_id(c->step_id);
      mc->set_action_count(c->action_count);
      if (timing_counter_) {
        mc->set_freed_at_count(c->in_use() ? 0 : c->freed_at_count);
      }
      h = c->next;
    }
  }

  mas->set_fragmentation_metric(GetFragmentation());

  // Record the recent size history
  uint64_t history_len = std::min(action_counter_, kMemDebugHistorySize);
  for (uint64_t i = action_counter_ - history_len; i < action_counter_; ++i) {
    SnapShot* ss = md.add_snap_shot();
    ss->set_action_count(i);
    uint64_t slot = i % kMemDebugHistorySize;
    ss->set_size(size_history_[slot]);
  }

  return md;
}

absl::optional<AllocatorStats> BFCAllocator::GetStats() {
  std::lock_guard<std::mutex> l(lock_);
  return stats_;
}

void BFCAllocator::ClearStats() {
  std::lock_guard<std::mutex> l(lock_);
  stats_.num_allocs = 0;
  stats_.peak_bytes_in_use = stats_.bytes_in_use;
  stats_.largest_alloc_size = 0;
}

std::array<BFCAllocator::BinDebugInfo, BFCAllocator::kNumBins>
BFCAllocator::get_bin_debug_info() {
  std::array<BinDebugInfo, kNumBins> bin_infos;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      BinNum bin_num = BinNumForSize(c->size);
      BinDebugInfo& bin_info = bin_infos[bin_num];
      bin_info.total_bytes_in_bin += c->size;
      bin_info.total_chunks_in_bin++;
      if (c->in_use()) {
        bin_info.total_bytes_in_use += c->size;
        bin_info.total_requested_bytes_in_use += c->requested_size;
        bin_info.total_chunks_in_use++;
      } else {
        Bin* bin = BinFromIndex(bin_num);
        CHECK_EQ(bin->free_chunks.count(h), 1);
        CHECK_EQ(c->bin_num, bin_num);
      }
      h = c->next;
    }
  }
  return bin_infos;
}
*/
}  // namespace cnindex
