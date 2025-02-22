'''
    Example how to get similar vectors from index giving a phrase
'''


from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import json
from utils.index_utils import extract_texts, extract_sentences_from_texts, encode_sentences
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import pipeline
import faiss
import pandas as pd
import sqlite3
import numpy as np


# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])



def main(argv):

    H = FLAGS.config

    device = H.run.device if torch.cuda.is_available() else -1

    ref_data = pd.read_csv(H.reference.path, sep="\t")
    ref_queries = ref_data.loc[:, "Query"].to_list()
    n_queries = len(ref_queries)
    ref_targets = ref_data.loc[:, [f"Target #{i}" for i in range(1, 6)]].to_dict(orient="list")

    #load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)

    #load index
    index = faiss.read_index(os.path.join(H.index.index_path,f"{H.index.index_name}.index"))

    #create or open db
    path_to_load_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_load_db)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot open database."

    cursor = connection.cursor()

    encodings = [np.zeros((1, H.data.len_embedding)).astype(np.float32) for _ in range(n_queries)]

    # Tokenize sentence
    inputs = tokenizer(ref_queries, return_tensors="pt", truncation=True, padding=True, max_length=H.model.model_max_length).to(device)

    # Get embeddings
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    target_outputs = {}
    for i in range(1, 6):
        target_outputs[i] = model(**tokenizer(ref_targets[f"Target #{i}"], return_tensors="pt", truncation=True, padding=True, max_length=H.model.model_max_length).to(device), output_hidden_states=True)
        with torch.inference_mode():
            target_outputs[i] = [j.reshape(1, -1) for j in target_outputs[i]['hidden_states'][-1][:,0,:].cpu().numpy().astype(np.float32)]
            for _idx in range(len(target_outputs[i])):
                target_encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)
                target_encoding[0] = target_outputs[i][_idx][0]
                target_outputs[i][_idx] = target_encoding
                faiss.normalize_L2(target_outputs[i][_idx])
        target_outputs[i] = np.concatenate(target_outputs[i], axis=0)

    # Use [CLS] token embedding as sentence encoding
    sentence_embedding = outputs['hidden_states'][-1][:,0,:].cpu().numpy().astype(np.float32)

    # encoding = sentence_embedding
    # Normalize
    for idx in range(len(encodings)):
        encodings[idx][0] = sentence_embedding[idx]
        faiss.normalize_L2(encodings[idx])

    output = {"queries": [], "corpus_results": [], "targets_results": []}
    for query, encoding in zip(ref_queries, encodings):

        query_target_results = []
        for i in range(1,6):
            # target_results = encoding @ target_outputs[i].T
            target_results = 1 - np.linalg.norm(encoding - target_outputs[i], axis=1) / 2
            assert target_results.shape == (n_queries,)
            target_i_results = []
            for idx, target in enumerate(ref_targets[f"Target #{i}"]):
                target_i_results.append({"target": target, "similarity": target_results[idx].item()})
            query_target_results.append(target_i_results)
        output["targets_results"].append(query_target_results)

        #search best matches in index
        distances, ann = index.search(encoding, k=H.retrieval.num_matches)
        query_corpus_results = []
        #retrieve best results from db
        output["queries"].append(query)
        for d, idx in zip(distances[0], ann[0]):
            d = 1 - d / 2
            cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id = {idx+1}")
            rows = cursor.fetchall()
            assert len(rows) == 1, "mmmh"
            query_corpus_results.append({"similarity": d.item(), "row": rows[0]})
            for row in rows:
                print(d, row)
                print("")

        output["corpus_results"].append(query_corpus_results)
        print("")

    with open(H.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    app.run(main)