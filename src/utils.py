# src/utils.py

import os
import ast
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
import joblib

def get_project_root():
    """Get the parent directory of 'src'."""
    return os.path.dirname(os.getcwd())

def load_embedding_model(embedding_path):
    """Load a word embedding model from the given path."""
    return KeyedVectors.load_word2vec_format(embedding_path, binary=True)

def safe_literal_eval(s):
    """Safely evaluate string to list (for symptom tokens)."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def embed_symptoms(symptom_list, embedding_model, embedding_dim=200):
    """Embed a list of symptoms into a single vector."""
    vectors = []
    for symptom in symptom_list:
        if symptom in embedding_model:
            vectors.append(embedding_model[symptom])
    if vectors:
        return np.mean(vectors, axis=0)  # Average vector
    else:
        return np.zeros(embedding_dim)   # If no symptoms found

def load_data(data_path):
    """Load symptom dataset and fix symptom_tokens."""
    df = pd.read_csv(data_path)
    df['symptom_tokens'] = df['symptom_tokens'].apply(safe_literal_eval)
    return df

def save_data(df, save_path):
    """Save dataframe to CSV."""
    df.to_csv(save_path, index=False)
    print(f"Embedded dataset saved at {save_path}")

def encode_labels(df, label_column):
    """Encode the label column and return updated df and the label encoder."""
    le = LabelEncoder()
    df[label_column + '_encoded'] = le.fit_transform(df[label_column])
    return df, le

def save_label_encoder(encoder, path):
    """Save the label encoder object for future use."""
    joblib.dump(encoder, path)
    print(f"Label encoder saved at {path}")