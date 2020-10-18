#   include <immintrin.h>
#   include <x86intrin.h>

// https://arxiv.org/pdf/1611.07612.pdf

__m256i count ( __m256i v) {
__m256i lookup =
_mm256_setr_epi8 (0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 ,
2 , 3 , 2 , 3 , 3 , 4 , 0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 ,
1 , 2 , 2 , 3 , 2 , 3 , 3 , 4) ;
__m256i low_mask = _mm256_set1_epi8 (0x0f ) ;
__m256i lo = _mm256_and_si256 (v, low_mask ) ;
__m256i hi = _mm256_and_si256 ( _mm256_srli_epi32
(v, 4) , low_mask ) ;
__m256i popcnt1 = _mm256_shuffle_epi8 (lookup ,
lo) ;
__m256i popcnt2 = _mm256_shuffle_epi8 (lookup ,
hi) ;
__m256i total = _mm256_add_epi8 ( popcnt1 , popcnt2
) ;
return _mm256_sad_epu8 (total ,
_mm256_setzero_si256 () ) ;
}
