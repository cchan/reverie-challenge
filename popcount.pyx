import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint8_t, int32_t, uint64_t, int64_t

cdef extern int __builtin_popcount(unsigned int) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _inplace_popcount_32(uint32_t[:] arr) nogil:
    cdef int i
    for i in xrange(arr.shape[0]):
        arr[i] = __builtin_popcount(arr[i])

def inplace_popcount_32(arr):
    _inplace_popcount_32(arr)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_popcount_bitwise_and(uint32_t[:] query_packed, uint32_t[:] fingerprints_packed) nogil:
    cdef int i
    for i in xrange(fingerprints_packed.shape[0]):
        fingerprints_packed[i] = __builtin_popcount(fingerprints_packed[i] & query_packed[i & (len(query_packed) - 1)])
        # Yes, the & (len(query_packed) - 1) is necessary. % len(query_packed) is way slower.

def fused_popcount_bitwise_and(query_packed, fingerprints_packed):
    _fused_popcount_bitwise_and(query_packed, fingerprints_packed)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_popcount_bitwise_and_notpacked_count(uint8_t[:] query_f, uint8_t[:,:] fingerprints, uint32_t[:] counts) nogil:
    cdef int i
    # assert(fingerprints.shape[1] == len(query_f) == 2048)
    for i in xrange(fingerprints.shape[0]):
        counts[i] = 0
        for j in xrange(2048):
            counts[i] += fingerprints[i][j] & query_f[j]

def fused_popcount_bitwise_and_notpacked_count(query_f, fingerprints):
    counts = np.zeros(fingerprints.shape[0], dtype=np.uint32)
    _fused_popcount_bitwise_and_notpacked_count(query_f, fingerprints, counts)
    return counts


cdef extern from "immintrin.h":
    ctypedef int __m256i
    __m256i _mm256_loadu_si256(__m256i *) nogil
    __m256i _mm256_cmpgt_epi8(__m256i, __m256i) nogil
    __m256i _mm256_setzero_si256() nogil
    int _mm256_movemask_epi8(__m256i) nogil
    int _mm_popcnt_u32(unsigned int) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_popcount_avx2(int32_t[::1] query_packed, uint8_t[:,:] fingerprints, uint32_t[:] counts) nogil:
    cdef:
        int i
        int j
        __m256i tmp
        int mask
    for i in xrange(fingerprints.shape[0]):
        counts[i] = 0
        for j in xrange(0, 64):
            tmp = _mm256_loadu_si256(<const __m256i*> &fingerprints[i][j<<5])
            tmp = _mm256_cmpgt_epi8(tmp, _mm256_setzero_si256())
            mask = _mm256_movemask_epi8(tmp)
            counts[i] += _mm_popcnt_u32(query_packed[j] & mask)

def fused_popcount_avx2(query_f, fingerprints):
    counts = np.zeros(fingerprints.shape[0], dtype=np.uint32)
    query_packed = np.frombuffer(np.packbits(query_f, bitorder='little'), dtype=np.int32)
    _fused_popcount_avx2(query_packed, fingerprints, counts)
    return counts


cdef extern from "immintrin.h":
    ctypedef int __int64
    __int64 _mm_popcnt_u64(__int64) nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_popcount64_bitwise_and(uint64_t[:] query_packed, uint64_t[:] fingerprints_packed, int32_t[:] counts) nogil: #, int64_t[:] counts) nogil:
    # cdef int i
    # cdef int j
    # for i in xrange(fingerprints_packed.shape[0]//32):
    #     for j in xrange(32):
    #         counts[i] += _mm_popcnt_u64(fingerprints_packed[i*32+j] & query_packed[j])
    cdef int i
    cdef int j
    cdef int32_t count
    cdef uint64_t *fingerprints_packed_curr
    for i in xrange(fingerprints_packed.shape[0]//32):
        count = 0
        fingerprints_packed_curr = &fingerprints_packed[i<<5]
        for j in xrange(32):
            count += _mm_popcnt_u64(fingerprints_packed_curr[j] & query_packed[j])
        counts[i] = count

def fused_popcount64_bitwise_and(query_packed, fingerprints_packed):
    counts = np.empty(fingerprints_packed.shape[0]//32, dtype=np.int32)
    _fused_popcount64_bitwise_and(query_packed, fingerprints_packed, counts)
    return counts



cdef extern from "immintrin.h":
    __m256i _mm256_and_si256(__m256i, __m256i) nogil
    __int64 _mm256_extract_epi64(__m256i, const int) nogil
    __m256i _mm256_add_epi64(__m256i, __m256i) nogil
    __m256i _mm256_setzero_si256() nogil
    __m256i _mm256_sad_epu8(__m256i, __m256i) nogil

cdef extern from "avx2_emulated_popcount.cpp":
    # uint64_t popcnt_AVX2_lookup(const uint8*, const uint32) nogil
    __m256i count ( __m256i ) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_avx2_emulated_popcount(uint64_t[:] query_packed, uint64_t[:] fingerprints_packed, uint32_t[:] counts) nogil:
    cdef:
        int i
        __m256i tmp
        __m256i acc
    for i in xrange(0, fingerprints_packed.shape[0], 32):
        acc = _mm256_setzero_si256()
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[0]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 0])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[4]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 4])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[8]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 8])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[12]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 12])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[16]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 16])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[20]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 20])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[24]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 24])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)
        tmp = _mm256_sad_epu8( count( _mm256_and_si256 (
                        _mm256_loadu_si256(<const __m256i*> &query_packed[28]),
                        _mm256_loadu_si256(<const __m256i*> &fingerprints_packed[i + 28])
                )), _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, tmp)

        counts[i>>5] += _mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 1) + \
                        _mm256_extract_epi64(acc, 2) + _mm256_extract_epi64(acc, 3)


def fused_avx2_emulated_popcount(query_packed, fingerprints_packed):
    counts = np.zeros(fingerprints_packed.shape[0], dtype=np.uint32)
    _fused_avx2_emulated_popcount(query_packed, fingerprints_packed, counts)
    return counts




cdef extern from "immintrin.h":
    cdef __m256i _mm256_shuffle_epi8(__m256i, __m256i) nogil
    cdef __m256i _mm256_add_epi8(__m256i, __m256i) nogil
    cdef __m256i _mm256_srli_epi16(__m256i, int) nogil
    cdef __m256i _mm256_set1_epi8(char) nogil
    cdef __m256i _mm256_setr_epi8(char, char, char, char, char, char, char, char, char, char, char, char, char, char, char, char,
                                  char, char, char, char, char, char, char, char, char, char, char, char, char, char, char, char) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_popcount64_bitwise_and_avx(uint64_t[:] query_packed, uint64_t[:] fingerprints_packed, int32_t[:] counts) nogil: #, int64_t[:] counts) nogil:
    # cdef int i
    # cdef int j
    # for i in xrange(fingerprints_packed.shape[0]//32):
    #     for j in xrange(32):
    #         counts[i] += _mm_popcnt_u64(fingerprints_packed[i*32+j] & query_packed[j])
    cdef int i
    cdef int j
    cdef int32_t count
    cdef uint64_t *fingerprints_packed_curr
    cdef __m256i vec
    cdef __m256i acc
    cdef __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2,
        1, 2, 2, 3,
        1, 2, 2, 3,
        2, 3, 3, 4,

        0, 1, 1, 2,
        1, 2, 2, 3,
        1, 2, 2, 3,
        2, 3, 3, 4
    )
    cdef __m256i low_mask = _mm256_set1_epi8(0x0f)
    cdef __m256i local
    cdef __m256i lo
    cdef __m256i hi
    for i in xrange(fingerprints_packed.shape[0]//32):
        local = _mm256_setzero_si256()
        fingerprints_packed_curr = &fingerprints_packed[i<<5]
        for j in xrange(0, 32, 4):
            vec = _mm256_and_si256(_mm256_loadu_si256(<const __m256i*>&fingerprints_packed_curr[j]), _mm256_loadu_si256(<const __m256i*>&query_packed[j]))
            lo = _mm256_and_si256(vec, low_mask)
            hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask)
            local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, lo))
            local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, hi))
        acc = _mm256_sad_epu8(local, _mm256_setzero_si256())
        counts[i] = _mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 1) + _mm256_extract_epi64(acc, 2) + _mm256_extract_epi64(acc, 3)

def fused_popcount64_bitwise_and_avx(query_packed, fingerprints_packed):
    counts = np.empty(fingerprints_packed.shape[0]//32, dtype=np.int32)
    _fused_popcount64_bitwise_and_avx(query_packed, fingerprints_packed, counts)
    return counts



cdef extern from "<algorithm>" namespace "std":
    Iter push_heap[Iter](Iter first, Iter last) nogil
    Iter pop_heap[Iter](Iter first, Iter last) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_popcount64_bitwise_and_avx_topk(uint64_t[:] query_packed, uint64_t[:] fingerprints_packed, int64_t[:] topk) nogil:
    cdef int i
    cdef int j
    cdef int k = len(topk)
    cdef int64_t count
    cdef uint64_t *fingerprints_packed_curr
    cdef int64_t heapentry
    cdef __m256i vec
    cdef __m256i acc
    cdef __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2,
        1, 2, 2, 3,
        1, 2, 2, 3,
        2, 3, 3, 4,

        0, 1, 1, 2,
        1, 2, 2, 3,
        1, 2, 2, 3,
        2, 3, 3, 4
    )
    cdef __m256i low_mask = _mm256_set1_epi8(0x0f)
    cdef __m256i local
    cdef __m256i lo
    cdef __m256i hi
    for i in xrange(fingerprints_packed.shape[0]//(2048//64)):
        local = _mm256_setzero_si256()
        fingerprints_packed_curr = &fingerprints_packed[i<<5]
        for j in xrange(0, 32, 4):
            vec = _mm256_and_si256(_mm256_loadu_si256(<const __m256i*>&fingerprints_packed_curr[j]), _mm256_loadu_si256(<const __m256i*>&query_packed[j]))
            lo = _mm256_and_si256(vec, low_mask)
            hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask)
            local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, lo))
            local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, hi))
        acc = _mm256_sad_epu8(local, _mm256_setzero_si256())
        count = _mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 1) + _mm256_extract_epi64(acc, 2) + _mm256_extract_epi64(acc, 3)
        # Maintaining a min-heap in topk (negated max heap, since stl is default max heap)
        heapentry = - ((count << 32) + i)
        if topk[0] > heapentry:
            pop_heap(&topk[0], &topk[k])
            topk[k-1] = heapentry
            push_heap(&topk[0], &topk[k])

def fused_popcount64_bitwise_and_avx_topk(query_packed, fingerprints_packed, k):
    topk = np.zeros(k, dtype=np.int64)
    _fused_popcount64_bitwise_and_avx_topk(query_packed, fingerprints_packed, topk)
    topk = -topk # negate it because minheap/maxheap stuff
    return topk >> 32, topk & np.int64((1<<32) - 1) # counts, indexes




cdef extern from "blosc.h":
    int blosc_getitem(void*, int, int, void*) nogil
    void blosc_cbuffer_sizes(void*, size_t*, size_t*, size_t*) nogil

cdef int32_t BLOCK_SIZE_IN_ROWS = 128*32 # 1MB blocks

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fused_popcount64_bitwise_and_avx_topk_blosc(uint64_t[:] query_packed, uint64_t[:] fingerprints_tmp_buf, const uint8_t[:] fingerprints_blosc_compressed, int64_t[:] topk, int n_fingerprints):
    cdef:
        int i
        int j
        int k = len(topk)
        int64_t count
        uint64_t *fingerprints_packed_curr
        int n_fingerprints_curr
        int z
        int64_t heapentry
        __m256i vec
        __m256i acc
        __m256i lookup = _mm256_setr_epi8(
            0, 1, 1, 2,
            1, 2, 2, 3,
            1, 2, 2, 3,
            2, 3, 3, 4,

            0, 1, 1, 2,
            1, 2, 2, 3,
            1, 2, 2, 3,
            2, 3, 3, 4
        )
        __m256i low_mask = _mm256_set1_epi8(0x0f)
        __m256i local
        __m256i lo
        __m256i hi
        size_t nbytes
        size_t cbytes
        size_t blocksize
    blosc_cbuffer_sizes(&fingerprints_blosc_compressed[0], &nbytes, &cbytes, &blocksize)
    print("INFO:", nbytes, cbytes, blocksize)
    assert(blocksize == BLOCK_SIZE_IN_ROWS*2048//8)
    for z in xrange(0, 32*n_fingerprints, 32*BLOCK_SIZE_IN_ROWS):
        n_fingerprints_curr = blosc_getitem(&fingerprints_blosc_compressed[0], z, min(32*BLOCK_SIZE_IN_ROWS, 32*n_fingerprints-z), &fingerprints_tmp_buf[0])
        if n_fingerprints_curr == 0:
            print("Something went wrong")
            break
        for i in xrange(n_fingerprints_curr//(2048//64)): # (2048//64) has to evenly divide n_fingerprints and len(fingerprints_tmp_buf) == BLOCK_SIZE_IN_ROWS
            local = _mm256_setzero_si256()
            fingerprints_packed_curr = &fingerprints_tmp_buf[i<<5]
            for j in xrange(0, 32, 4):
                vec = _mm256_and_si256(_mm256_loadu_si256(<const __m256i*>&fingerprints_packed_curr[j]), _mm256_loadu_si256(<const __m256i*>&query_packed[j]))
                lo = _mm256_and_si256(vec, low_mask)
                hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask)
                local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, lo))
                local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, hi))
            acc = _mm256_sad_epu8(local, _mm256_setzero_si256())
            count = _mm256_extract_epi64(acc, 0) + _mm256_extract_epi64(acc, 1) + _mm256_extract_epi64(acc, 2) + _mm256_extract_epi64(acc, 3)
            # Maintaining a min-heap in topk (negated max heap, since stl is default max heap)
            heapentry = - ((count << 32) + (i + z//32))
            if topk[0] > heapentry:
                pop_heap(&topk[0], &topk[k])
                topk[k-1] = heapentry
                push_heap(&topk[0], &topk[k])

def fused_popcount64_bitwise_and_avx_topk_blosc(query_packed, fingerprints_blosc_compressed, k, n_fingerprints):
    topk = np.zeros(k, dtype=np.int64)
    fingerprints_tmp_buf = np.empty(32*BLOCK_SIZE_IN_ROWS, dtype=np.uint64) # 128 => 64kB block
    _fused_popcount64_bitwise_and_avx_topk_blosc(query_packed, fingerprints_tmp_buf, fingerprints_blosc_compressed, topk, n_fingerprints)
    topk = -topk # negate it because minheap/maxheap stuff
    return topk >> 32, topk & np.int64((1<<32) - 1) # counts, indexes
