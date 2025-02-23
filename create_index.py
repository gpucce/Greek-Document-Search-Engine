
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import oimdp
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import json
from utils.index_utils import (
    parse_tlg_dataset_folder_structure,
    parse_openiti_dataset_folder_structure,
    parse_hadith_dataset_folder_structure,
    extract_tlg_texts,
    extract_openiti_texts,
    extract_hadith_texts,
    extract_sentences_from_texts,
    encode_sentences
)

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import oimdp
from transformers import pipeline
import faiss
import pandas as pd
import sqlite3

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def main(argv):
    H = FLAGS.config

    device = H.run.device if torch.cuda.is_available() else -1

    #path to json dataset
    json_dataset_path = H.data.json_dataset_path


    #load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)
    model.resize_token_embeddings(len(tokenizer))
    mask_filler = pipeline("fill-mask", model = H.model.model, tokenizer = H.model.tokenizer, top_k = H.model.top_k, device=device)

    #create index
    path_to_save_index = os.path.join(H.index.index_path,f"{H.index.index_name}.index")
    # Controlla se il file dell'indice esiste già
    if os.path.exists(path_to_save_index):
        # Carica l'indice esistente
        index = faiss.read_index(path_to_save_index)
        print("Index successfully loaded!")
    else:
        # Se non esiste, crea un nuovo indice
        index = faiss.IndexFlatL2(H.data.len_embedding)
        print("New index created.")

    #create or open db
    path_to_save_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_save_db)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot create or open database."

    cursor = connection.cursor()
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {H.db.db_name} (row_id INTEGER PRIMARY KEY AUTOINCREMENT, author_id TEXT, id TEXT, name TEXT, sentence TEXT, citations TEXT, book_name TEXT)")

    if H.data.dataset_type == "greek_tlg":
        data_path_generator = parse_tlg_dataset_folder_structure(json_dataset_path)
    elif H.data.dataset_type == "arabic_openiti":
        data_path_generator = parse_openiti_dataset_folder_structure(json_dataset_path)
    elif H.data.dataset_type == "hadith":
        data_path_generator = parse_hadith_dataset_folder_structure(json_dataset_path)
    else:
        raise ValueError(f"Invalid dataset type {H.data.dataset_typ}.")

    for file_path, file_name, folder_name, num_current_folder in data_path_generator:
        # Check if is a JSON file
        print(f"    JSON: {file_name}")

        # Leggi il contenuto del file JSON
        if H.data.dataset_type == "greek_tlg":
            with open(file_path, "r", encoding="utf-8") as json_file:
                #get json data
                data = json.load(json_file)
                #extract all text fields
            texts, citations = extract_tlg_texts(data)

        elif H.data.dataset_type == "arabic_openiti":
            with open(file_path, "r", encoding="utf-8") as oimdp_file:
                data = oimdp.parse(oimdp_file.read())
            texts, citations = extract_openiti_texts(data)

        elif H.data.dataset_type == "hadith":
            with open(file_path, "r", encoding="utf-8") as oimdp_file:
                data = oimdp.parse(oimdp_file.read())
            texts, citations = extract_hadith_texts(data)

        else:
            raise ValueError(f"Invalid dataset type {H.data.dataset_typ}.")

        #join texts into a single sentence if they semantically belong together
        sentences, citations = extract_sentences_from_texts(texts, citations, mask_filler, H.data.min_words_in_phrase, H.model.model_max_length, tokenizer )

        #get encoding of each sentence
        sentence_embeddings = encode_sentences(sentences, model, tokenizer, H.data.len_embedding, device, H.model.model_max_length)

        #add embeddings to index
        faiss.normalize_L2(sentence_embeddings)
        index.add(sentence_embeddings)

        #save data to db (NB: if FAISS USE INDEX K FOR A SENTENCE, SQLITE USE INDEX (K+1))
        for idx, sentence in enumerate(sentences):
                cursor.execute(f"""
                                    INSERT INTO {H.db.db_name} (author_id, id, name, sentence, citations, book_name)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (data['author_id'], data['id'], data['name'], sentence, str(citations[idx]), folder_name))
        connection.commit()
        #save index
        faiss.write_index(index, path_to_save_index)


    connection.close()



if __name__ == '__main__':
    app.run(main)