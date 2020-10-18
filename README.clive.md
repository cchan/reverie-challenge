# Clive's solution README

I used a Jupyter notebook `performance.ipynb` for my experimentation, and exported the final result to `app.py`. `popcount.pyx` contains code for several iterations of the Cython acceleration code. `avx2_emulated_popcount.cpp` is part of unfinished attempt at getting faster popcounts to work using AVX2, which is documented in `performance.ipynb` too.

See the notebook for answers to questions 1, 2, 3, and a full walkthrough of the design process.

As for 4:

```
conda install -y cython
cython popcount.pyx
gcc -Wall -O3 -g -lm -shared -pthread -fPIC -fwrapv -fno-strict-aliasing -mavx2 -fopt-info-vec-optimized -I$CONDA_PREFIX/include/python3.6m -I$CONDA_PREFIX/include -I$CONDA_PREFIX/lib/python3.6/site-packages/numpy/core/include/ -o popcount.so popcount.c
streamlit run app.py
```

should get the app up and running.
