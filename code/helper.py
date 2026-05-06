import pandas as pd
import numpy as np
from pathlib import Path


def prepare_symptoms_array(symptoms):
    '''
    Convert a list of symptoms to a ndim(X) (in this case 131) that matches the
    dataframe used to train the machine learning model

    Output:
    - X (np.array) = X values ready as input to ML model to get prediction
    '''
    symptoms_array = np.zeros((1, 133))
    
    # Get path relative to this file
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "data" / "clean_dataset.tsv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")
    
    df = pd.read_csv(data_file, sep='\t')
    
    for symptom in symptoms:
        symptom_idx = df.columns.get_loc(symptom)
        symptoms_array[0, symptom_idx] = 1

    return symptoms_array