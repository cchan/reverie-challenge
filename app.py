import streamlit as st
import numpy as np
import pandas as pd
import time

from starter import compute_fingerprint, load_database
from rdkit import Chem

from popcount import fused_popcount_avx2

@st.cache(suppress_st_warning=True)
def load():
    with st.spinner("Loading DB..."):
        return load_database("database_fingerprints.npy"), pd.read_csv("database.csv")

fingerprints, molecules = load()
st.success("DB loaded")

query = st.text_input("Enter SMILES:", "C=CC(=O)N1CCCC(C1)N2C3=NC=NC(=C3C(=N2)C4=CC=C(C=C4)OC5=CC=CC=C5)N")
k = st.number_input("How many top results to return:", 5)

if Chem.MolFromSmiles(query) is None:
    st.write("Invalid SMILES")
else:
    start = time.perf_counter()
    query_f = compute_fingerprint(query).astype(np.uint8)
    scores = fused_popcount_avx2(query_f, fingerprints) / np.sum(query_f)
    topk = np.argpartition(-scores, k)[:k]
    df = molecules.loc[topk,["smiles","ID"]]
    df["coverage_score"] = scores[topk]
    interval = time.perf_counter() - start
    st.write(df.sort_values("coverage_score", ascending=False))
    st.write("Returned",k,"results in",interval,"sec")
