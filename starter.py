import numpy as np
from rdkit import Chem


def compute_fingerprint(smiles_string: str) -> np.array:
    """
    Computes the RDK Fingerprint.
    More info: http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html
    Args:
        smiles_string: input molecule, represetned as a SMILES
    Returns:
        descriptor of molecule as a binary 1D np.array
    """
    mol = Chem.MolFromSmiles(smiles_string)
    try:
        return np.array(
            list(map(int, (Chem.RDKFingerprint(mol, fpSize=2048, maxPath=5).ToBitString())))
        )
    except Exception:
        return None

def load_database(db_file: str) -> np.array:
	""" Load database of molecules to compare to.

	Args:
		db_file: path to a .npy file containing molecular fingerprints
	Returns:
		np.array of the loaded fingerprints 
	"""

	db = np.load(db_file)
	return db
