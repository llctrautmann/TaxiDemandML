# paths.py is a meta file that defines the paths for all the data and the files that are being used 
# in the ML project. This is being done because it would be cumbersome to copy and paste the correct
# file paths into every notebook of python file. In addition, maintaining it would also be a nightmare
# so we avoid that with the paths.py file.

from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data' 
RAW_DATA_DIR = PARENT_DIR / 'data' / 'raw'
CLEANED_DATA_DIR = PARENT_DIR / 'data' / 'cleaned'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(CLEANED_DATA_DIR).exists():
    os.mkdir(CLEANED_DATA_DIR)