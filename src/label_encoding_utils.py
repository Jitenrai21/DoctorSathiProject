# src/label_encoding_utils.py

import os
from utils import (
    get_project_root,
    load_data,
    encode_labels,
    save_data,
    save_label_encoder
)

def main():
    project_root = get_project_root()

    # Paths
    embedded_data_path = os.path.join(project_root, "data", "embedded_symptom_dataset.csv")
    encoded_data_path = os.path.join(project_root, "data", "encoded_symptom_dataset.csv")
    label_encoder_path = os.path.join(project_root, "models", "label_encoder.pkl")

    # Load embedded symptom dataset
    df = load_data(embedded_data_path)

    # Encode disease labels
    df_encoded, label_encoder = encode_labels(df, label_column='label')

    # Save encoded dataset
    save_data(df_encoded, encoded_data_path)

    # Save the label encoder
    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
    save_label_encoder(label_encoder, label_encoder_path)

if __name__ == "__main__":
    main()