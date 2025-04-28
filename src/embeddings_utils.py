# src/embeddings_utils.py

import os
from utils import (
    get_project_root,
    load_embedding_model,
    load_data,
    embed_symptoms,
    save_data
)

def main():
    project_root = get_project_root()

    # Paths
    embedding_path = os.path.join(project_root, "data", "embeddings", "BioWordVec_PubMed_MIMICIII_d200.vec.bin")
    data_path = os.path.join(project_root, "data", "processed_symptom_dataset.csv")
    embedded_data_path = os.path.join(project_root, "data", "embedded_symptom_dataset.csv")

    # Load embedding model and data
    bio_word_vec = load_embedding_model(embedding_path)
    df_cleaned = load_data(data_path)

    # Apply embeddings
    df_cleaned['symptom_vector'] = df_cleaned['symptom_tokens'].apply(
        lambda x: embed_symptoms(x, bio_word_vec)
    )

    # Save the embedded dataset
    save_data(df_cleaned, embedded_data_path)

if __name__ == "__main__":
    main()