import gradio as gr
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import json
from utils.launcher_utils import get_best_results
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import pipeline
import faiss
import pandas as pd
import sqlite3
import numpy as np
import ast


# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "/home/giuliofederico/Itserr/config/index_config.py", "configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

import unicodedata
import re

from bs4 import BeautifulSoup

SHOW_CIT = False
SHOW_MATCH = False

def highlight_words_in_html(html, parole, stile):
    soup = BeautifulSoup(html, "html.parser")  
    
    for parola in parole:
        # Trova e sostituisci solo il testo (non i tag)
        for elem in soup.find_all(string=re.compile(r'\b' + re.escape(parola) + r'\b')):
            nuovo_contenuto = re.sub(
                r'\b' + re.escape(parola) + r'\b', 
                f'<span style="{stile}">{parola}</span>', 
                elem
            )
            elem.replace_with(BeautifulSoup(nuovo_contenuto, "html.parser"))
    
    return str(soup)

def normalize_text(text):
    # Rimuovere caratteri invisibili come spazi non separabili
    text = text.replace('\xa0', ' ')  # Sostituire \xa0 con uno spazio
    # Normalizzazione Unicode
    text = unicodedata.normalize('NFKC', text)
    # Rimuovere eventuali spazi extra all'inizio e alla fine
    text = text.strip()
    # Rimuovere caratteri invisibili come nuove righe e tabulazioni
    text = re.sub(r'\s+', ' ', text)  # Sostituire sequenze di spazi con un singolo spazio
    return text

def main(argv):

    H = FLAGS.config

    #load list of works
    list_of_works = np.load(H.data.name_of_the_works_path)

    #TODO in the future change DB and index based on dropdown option
    device = H.run.device if torch.cuda.is_available() else -1

    #load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)

    #load index
    index = faiss.read_index(os.path.join(H.index.index_path,f"{H.index.index_name}.index"))

    #create or open db
    path_to_load_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_load_db, check_same_thread=False)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot open database."

    cursor = connection.cursor()


    def process_inputs(text, number, max_documents, option, author_id, works_selected, additional_text, additional_text_slider_value):


        max_documents = max(number, max_documents)
        #print(f"SELEZIONE: text: {text},  number: {number},  option: {option},  author_id: {author_id},  works_selected: {works_selected},  additional_text: {additional_text},  slider_value: {additional_text_slider_value}")

        if text=="":
            return f"""
            <div style="border: 2px solid #ccc; padding: 10px; margin-bottom: 10px; background-color: white;">
                <p>Insert a valid query.</p>
            </div> """
        
        best_results = get_best_results(index, H, cursor, text, tokenizer, model, number, max_documents, author_id, works_selected, additional_text, additional_text_slider_value, device )
        #print(best_results)
        results = []
        for i in range(len(best_results)):
            result = {
                "Book": best_results[i]['book_name'],
                "author_id": best_results[i]['author_id'],
                "id": best_results[i]['id'],
                "name": best_results[i]['name'],
                "match": best_results[i]['sentence'],
                "citations": best_results[i]['citations']
            }
            results.append(result)
        
        
        # Creare una stringa HTML per visualizzare tutti i risultati
        results_html = ""
        for result in results:
            match_text = result['match']
           
            if SHOW_CIT:
                # Normalizzare match_text rimuovendo i caratteri speciali
                match_text = normalize_text(match_text)

                # Convertire la stringa 'citations' in una lista di dizionari
                if result['citations']:
                    try:
                        citations = ast.literal_eval(result['citations'])
                        placeholders = {}  # Dizionario per tracciare i segnaposto
                        placeholder_template = "__CITATION_PLACEHOLDER_{}__"

                        for i, citation in enumerate(citations):
                            citation_text = citation.get('text')
                            if citation_text:
                                citation_id = citation['citation']
                                citation_text = normalize_text(citation_text)
                                if citation_text in match_text:
                                    placeholder = placeholder_template.format(i)
                                    placeholders[placeholder] = f'<span class="citation" data-citation="{citation_id}" title="{citation_text}">{citation_text}</span>'
                                    match_text = match_text.replace(citation_text, placeholder)

                        # Sostituisci i segnaposto con il codice HTML finale
                        for placeholder, html in placeholders.items():
                            match_text = match_text.replace(placeholder, html)


                                    
                    except json.JSONDecodeError:
                        print(f"Error decoding quotes: {result['citations']}")
            
            if SHOW_MATCH:
                #evidenzia nei risultati tutte le parole comuni con la query
                words_text = re.findall(r'\b\w+\b', text)  # Converting to lowercase per confronti insensibili al maiuscolo/minuscolo

                #Controllare se ciascuna parola della query è presente nel test matchato
                words_match_text = re.findall(r'\b\w+\b', match_text)  

                #Troviamo le parole che si trovano in entrambe le stringhe
                common_words = [word for word in words_text if word in words_match_text]

                stile_css = "font-weight: bold; color: blue;"

                match_text = highlight_words_in_html(match_text, common_words, stile_css)


            # Creare una box per ogni risultato
            results_html += f"""
            <div style="border: 2px solid #ccc; padding: 10px; margin-bottom: 10px; color: black">
                <strong>Book:</strong> {result['Book']}<br>
                <strong>author_id:</strong> {result['author_id']}<br>
                <strong>id:</strong> {result['id']}<br>
                <strong>name:</strong> {result['name']}<br>
                <strong>match:</strong> {match_text}<br>
            </div>
            """

        # Aggiungere il CSS per l'effetto hover
        results_html += """
        <style>
            .citation {
                color: rgb(1, 3, 39);  /* Cambia il colore al passaggio del mouse */
                text-decoration-line: underline;  /* Opzionale: rendere il testo più evidente */

            }

            /* Aggiungere un effetto al passaggio del mouse */
            .citation:hover {
                
                font-weight: bold;  /* Opzionale: rendere il testo più evidente */
                background-color: #e6e6e6;  /* Un piccolo effetto di sfondo per l'hover */
            }

            /* Aggiungi un effetto di tooltip che appare al passaggio del mouse */
            .citation[data-citation]:hover::after {
                content: attr(data-citation);  /* Mostra il contenuto del tooltip */
                position: absolute;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 5px;
                border-radius: 5px;
                font-size: 12px;
                white-space: nowrap;
                z-index: 9999;
            }
                    </style>
        """
        
        return results_html


    # Funzione di filtro per le opere
    def filter_works(query):
        query = query.lower()
        return [opera for opera in list_of_works if query in opera.lower()]

    # Layout UI con un campo di ricerca per le opere
    demo = gr.Interface(
        fn=process_inputs,
        inputs=[
            gr.Textbox(lines=5, 
                        placeholder="Servius ad Virgil. Aen. III, 334: [Chaonios cognomine Campos] Epirum campos non habere omnibus notum est.",
                        label="Enter query"),
            gr.Slider(1, 10, step=1, label="Choose how many results to return"),
            gr.Slider(1, 1000, step=1, label="How many results to search before returning the best choices (above), important if author/work is specified."),
            gr.Dropdown(["DB_Greek"], label="Select the database where to search"),
            gr.Textbox(label="Author ID", placeholder="Enter the author ID (numeric)"),
            gr.Dropdown(choices=list(np.insert(list_of_works, 0, 'All')), label="Select Works", multiselect=False),  # Lista di opere
            gr.Textbox(label="Additional Phrase", placeholder="Enter an additional phrase"),
            gr.Slider(0, 1, step=0.01, label="How much weight should be given to the additional sentence compared to the main one.")
        ],
        outputs=gr.HTML(),
        title="Greek Document Search Engine",
        description="Enter a text query and the number of results you want to get. The system will search the documents for the best results and automatically sort them.",
    )

    demo.launch(server_name="0.0.0.0", server_port=40001, share=False)

if __name__ == '__main__':
    app.run(main)