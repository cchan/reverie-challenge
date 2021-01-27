import time
class Timer:
    def __init__(self, s, iters):
        self.s = s
        self.iters = iters
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        print("\"" + self.s + "\"", "took", (time.perf_counter() - self.start)/self.iters*1e9, "nanoseconds per db row")


import numpy as np
import pandas as pd

from starter import compute_fingerprint, load_database

with Timer("db loading", 1000000):
    fingerprints = load_database("database_fingerprints.npy")
    molecules = pd.read_csv("database.csv")
with open("query.txt") as q:
  query = q.readline()


k = 5


import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray, to_gpu


with Timer("pre-packing process", 1000000):
    fingerprints_packed_gpu = to_gpu(np.frombuffer(np.packbits(fingerprints, axis=1, bitorder='big'), dtype=np.uint64))
    print(fingerprints_packed_gpu.shape)
    sums_gpu = GPUArray((1000000,), dtype=np.int32)
    n_fingerprints = len(fingerprints)


mod = SourceModule("""
  __global__ void and_popcount_topk(unsigned long long *a, unsigned long long *b, int *sums)
  {
    int ME = blockIdx.x * blockDim.x + threadIdx.x;
    int START = ME * 1737;
    int END = min((ME + 1) * 1737, 1000000);
    int topk["""+str(k)+"""];
    for (int i = START; i < END; i++) {
      int sum = 0;
      unsigned long long *a_tmp = &a[i<<5];
      for (int j = 0; j < 32; j++) {
        sum += __popcll(a_tmp[j] & b[j]);
      }
      //if(sum > topk[0]) {
      //  bubble it
      //}
      sums[i] = sum;
    }
    // Place sums in global memory
    // Global thread sync
    // NOW... merge in log time.
    // Or... just merge in two stages? Merge, sync, merge.
    int pow = 1;
    for(; ME & pow == 0; pow <<= 1) {
    //    merge entries from [ME] and [ME | pow]
    //    synchronize
    }
  }
  """, options=['--use_fast_math', '-O3', '-Xptxas', '-O3,-v'])
and_popcount_topk = mod.get_function("and_popcount_topk")


from rdkit import Chem
mol = Chem.MolFromSmiles(query)
s = Chem.RDKFingerprint(mol, fpSize=2048, maxPath=5).ToBitString()
query_packed_gpu = to_gpu(np.frombuffer(int(s, 2).to_bytes(len(s) // 8, byteorder='big'), dtype=np.uint64))
print(query_packed_gpu.shape)


with Timer("popcount", 1000000000):
    for i in range(1000):
        # do a prepared call here instead
        and_popcount_topk(fingerprints_packed_gpu, query_packed_gpu, sums_gpu, grid=(18,1,1), block=(32,1,1))
print(sums_gpu[5:])
#print(pycuda.gpuarray.sum(sums_gpu))
