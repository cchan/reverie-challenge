import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint8_t, int32_t, uint64_t

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
cdef void _fused_popcount64_bitwise_and(uint64_t[:] query_packed, uint64_t[:] fingerprints_packed, uint32_t[:] counts) nogil:
    cdef int i
    for i in xrange(fingerprints_packed.shape[0]):
        counts[i>>5] += _mm_popcnt_u64(fingerprints_packed[i] & query_packed[i & 31])

def fused_popcount64_bitwise_and(query_packed, fingerprints_packed):
    counts = np.zeros(fingerprints_packed.shape[0], dtype=np.uint32)
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
