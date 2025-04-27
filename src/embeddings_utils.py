import os
import ast
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

# Get the parent of 'src'
project_root = os.path.dirname(os.getcwd())

# Create the correct path
embedding_path = os.path.join(project_root, "data", "embeddings", "BioWordVec_PubMed_MIMICIII_d200.vec.bin")

# Load the model
bio_word_vec = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

def embed_symptoms(symptom_list, embedding_model, embedding_dim=200):
    vectors = []
    for symptom in symptom_list:
        if symptom in embedding_model:
            vectors.append(embedding_model[symptom])
    if vectors:
        return np.mean(vectors, axis=0)  # Average vector
    else:
        return np.zeros(embedding_dim)   # If no known symptoms
    

data_path = os.path.join(project_root, "data", "processed_symptom_dataset.csv")

data = pd.read_csv(data_path)

df_cleaned = pd.DataFrame(data)

# FIX: Convert symptom_tokens from string to list
# (If it's already a list, this does nothing. If stored as string, it will fix.)
df_cleaned['symptom_tokens'] = df_cleaned['symptom_tokens'].apply(ast.literal_eval)

# Apply embeddings
df_cleaned['symptom_vector'] = df_cleaned['symptom_tokens'].apply(
    lambda x: embed_symptoms(x, bio_word_vec)
)

# Save the embedded dataset
embedded_data_path = os.path.join(project_root, "data", "embedded_symptom_dataset.csv")
df_cleaned.to_csv(embedded_data_path, index=False)

print(f"Embedded dataset saved at {embedded_data_path}")