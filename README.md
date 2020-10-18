# Reverie Labs Software Engineer Coding Challenge

## Deliverables
1. You should have a coverage scoring function with the following signature (provided in Python, please adapt if you use a different programming language)

``` python
def molecular_coverage(query: str, reference: str) -> float:
    """ Computes the substructure coverage of the query compound by the reference compound
    
    Args:
        query: A SMILES representation of the query compound
        reference: A SMILES representation of the reference compound
    Returns:
        A score indicating the coverage of substructures in the query that exist in the reference
    """
    pass
```

2. A script that runs substructure coverage scoring on databases of compounds. 
Given the following:
- a file `query.txt`, which has a single line containing a SMILES string for a query compound
- a `database_fingerprints.npy` file containing molecular fingerprints of a database of reference compounds (potentially up to a million)
- a `database.csv` file containing the SMILES strings associated with the `.npy` file.
- an integer `K`

Your script should output the `K` most covered compounds in `database.csv` to `query.txt`. Your script should be able to handle an input that looks something like this:
``` bash
./molcoverage --query query.txt --fingerprints database_fingerprints.npy --source_smiles database.csv --K 10 --out results.txt
```  
and output `results.txt` with the SMILES strings of the `K` most covered compounds and their coverage score.

As an extension if you have time, handle the case where `query.txt` contains `M > 1` SMILES strings (1 per line). In this case, your script should return the most covered `K` compounds to each of the `M` compounds in `query.txt`.

3. Unit Tests: Write unit tests that verify that your code is working. We're leaving this section open-ended - show us your experience with writing tests. 

4. A simple Web UI: Build a simple UI (a browser-based app) that interfaces with your script. The application should allow a user to enter in a SMILES string and `K` and get a list of the `K` most covered compounds in the set. We strongly recommend using a framework like Streamlit to rapidly build this. We don't expect a perfect UI - we want to see your rapid prototyping skills.


## Setting up your environment.
We strongly recommend you use Docker for this coding challenge. You can install Docker using instructions [here](https://docs.docker.com/get-docker/). If you have Docker installed, you should be able to build a container with all of the files you need installed by running

```
docker build -t rdkit_build . 
```
This will use the `Dockerfile` in this repo to build a container. It will install conda/pip packages using the provided `environment.yml` file. You should feel free to edit this file with additional deps you need and rebuild.

Once your container is built, you can run
``` bash
docker run -it -v $(pwd):/src rdkit_build
conda activate myenv
```
This will allow you to circumvent having to install Anaconda and every dependency on your machine.

If you'd prefer to not use Docker, you can prepare your environment manually by [installing Anaconda](https://docs.conda.io/en/latest/miniconda.html), and then running
```
conda env create -f environment.yml
conda activate myenv
```
Note that if you use the docker-based setup, you don't need to run this step. It's specified in the Dockerfile for you. 

## Starter Code
`starter.py` contains helper functions that you can use in this challenge. 
Here are some examples:

### Computing the molecular fingerprint of a molecule.

``` python
from starter import compute_fingerprint
my_smiles_string = "CCC" # We provide you with a million examples of these.
fingerprint = compute_fingerprint(my_smiles_string)
```

### Loading the database of molecular fingerprints
``` python
from starter import load_database
db = load_database("database_fingerprints.npy")
db = load_database(db_file)
```

### Verifying that the provided fingerprints are correct
You can combine the above two steps the verify that the fingerprints we provide you are correct.
For example, to check the first 1000:
``` python
from starter import compute_fingerprint, load_database

import numpy as np
import pandas as pd

df = pd.read_csv("database.csv")
db = load_database("database_fingerprints.npy")
for i in range(1000):
    fingerprint = compute_fingerprint(df["smiles"][i])
    assert(np.array_equal(fingerprint, db[i]))
```
