from  pathlib import Path

#Ce fichier permet de centraliser et construire proprement les chemins de ton projet, 
#afin que ton code continue de fonctionner même si tu déplaces le projet ou changes de machine.


PROJECT_ROOT = Path(__file__).parent.parent


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR  / "processed"

