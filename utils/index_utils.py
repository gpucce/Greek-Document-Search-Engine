import re
import os
import torch
import numpy as np
import oimdp

from arabert.preprocess import ArabertPreprocessor
model_name="bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)


def parse_tlg_dataset_folder_structure(json_dataset_path):
    num_current_folder = 1
    for folder_name in os.listdir(json_dataset_path):
        folder_path = os.path.join(json_dataset_path, folder_name)

        # check if  is a directory
        if os.path.isdir(folder_path):
            print(f"[{num_current_folder}/{len(os.listdir(json_dataset_path))}] Author: {folder_name}")

            # Iterate on each json file
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith(".json"):
                    yield file_path, file_name, folder_name, num_current_folder
            num_current_folder += 1

def extract_tlg_texts(data):
    texts = []
    citations = []

    for entry in data["content"]:
        if "text" in entry and entry['text'].strip() != "":
            texts.append(entry["text"])

            if 'citation' in entry:
                entry['text'] = entry['text']
                citations.append(entry)
            else:
                citations.append("")


    return texts, citations

def parse_hadith_dataset_folder_structure(json_dataset_path):
    num_current_folder = 1
    for folder_name in sorted(os.listdir(json_dataset_path)):
        folder_path = os.path.join(json_dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        sorted_paths = sorted(os.listdir(folder_path))

        # check if  is a directory
        if sorted_paths:
            print(f"[{num_current_folder}/{len(os.listdir(json_dataset_path))}] Author: {folder_name}")

            # Iterate on each json file
            for second_folder_name in sorted_paths:
                second_folder_path = os.path.join(folder_path, second_folder_name)
                if os.path.isfile(second_folder_path):
                    if (second_folder_path.endswith("ara1") or second_folder_path.endswith("ara1.mARkdown")) or (second_folder_path.endswith("ara1") or second_folder_path.endswith("ara1.txt")):
                        yield second_folder_path, second_folder_name, folder_name, num_current_folder
                    continue
                sorted_second_paths = sorted(os.listdir(second_folder_path))

                for third_folder_name in sorted_second_paths:
                    third_folder_path = os.path.join(second_folder_path, third_folder_name)
                    if not os.path.isdir(third_folder_path):
                        continue
                    third_paths = [x for x in os.listdir(third_folder_path)]
                    sorted_third_paths = sorted(third_paths, key=lambda x: int(x[:4]))

                    for file_name in sorted_third_paths:
                        file_path = os.path.join(third_folder_path, file_name)
                        yield file_path, file_name, folder_name, num_current_folder
            num_current_folder += 1

def parse_openiti_dataset_folder_structure(json_dataset_path):
    num_current_folder = 1
    for folder_name in sorted([i for i in os.listdir(json_dataset_path) if i != "fonti_arabo_wp8"], key=lambda x: int(x[:4])):
        folder_path = os.path.join(json_dataset_path, folder_name, "data")
        sorted_paths = sorted(os.listdir(folder_path), key=lambda x: int(x[:4]))

        # check if  is a directory
        if sorted_paths:
            print(f"[{num_current_folder}/{len(os.listdir(json_dataset_path))}] Author: {folder_name}")

            # Iterate on each json file
            for second_folder_name in sorted_paths:
                second_folder_path = os.path.join(folder_path, second_folder_name)
                sorted_second_paths = sorted(os.listdir(second_folder_path), key=lambda x: int(x[:4]))

                for third_folder_name in sorted_second_paths:
                    third_folder_path = os.path.join(second_folder_path, third_folder_name)
                    if not os.path.isdir(third_folder_path):
                        continue
                    third_paths = [x for x in os.listdir(third_folder_path) if x.endswith("ara1") or x.endswith("ara1.mARkdown")]
                    sorted_third_paths = sorted(third_paths, key=lambda x: int(x[:4]))

                    for file_name in sorted_third_paths:
                        file_path = os.path.join(third_folder_path, file_name)
                        yield file_path, file_name, folder_name, num_current_folder
            num_current_folder += 1


def postprocess_openiti(text):
    if "Vol." in text:
        text = text.split("Vol.")[0]
    text = arabert_prep.preprocess(text)
    return text.strip()

def extract_hadith_texts(data):
    out = []
    lines = []
    for text_chunk in data.content:
        if not isinstance(text_chunk, oimdp.structures.Paragraph):

            try:
                lines.append(str(text_chunk))
            except:
                continue

        else:
            if lines:
                out.append(" ".join(lines))
                lines = []
    if lines:
        out.append(" ".join(lines))

    out = [postprocess_openiti(i) for i in out]
    return out, None

def extract_openiti_texts(data):
    raise NotImplementedError("This function is not implemented yet")

def merge_sentences_with_mask(sentence1, sentence2, tokenizer, with_mask=True, add_space_between_sentences=False,):

    mask_token = tokenizer.mask_token

    # remove spaces
    sentence1 = sentence1.strip()
    sentence2 = sentence2.strip()

    # get last word of the first sentence
    words1 = sentence1.split()
    last_word1 = words1.pop()

    # get first word of the second sentence
    words2 = sentence2.split()
    first_word2 = words2.pop(0)

    # combine (masking second word) the two sentences
    if with_mask:
        if add_space_between_sentences:
            merged_sentence = " ".join(words1) + " " + last_word1 + " " + f"{mask_token} " + " ".join(words2)
        else:
            merged_sentence = " ".join(words1) + " " + last_word1 + f"{mask_token} " + " ".join(words2)
    # combine (using second word) the two sentences
    else:
        if add_space_between_sentences:
            merged_sentence = " ".join(words1) + " " + last_word1 + " " + f"{first_word2} " + " ".join(words2)
        else:
            merged_sentence = " ".join(words1) + " " + last_word1 + f"{first_word2} " + " ".join(words2)

    return merged_sentence


def check_if_two_words_go_together(current_sentence, current_text, mask_filler, max_lenght, tokenizer):

    #create a unique sentence using current sentence and text masking first word of current text
    masked_sentence = merge_sentences_with_mask(current_sentence, current_text, tokenizer)
    joined_sentence = merge_sentences_with_mask(current_sentence, current_text, tokenizer, with_mask=False)


    tokens = tokenizer(
        masked_sentence,
        truncation=True,
        max_length=max_lenght,
        return_tensors="pt",
        add_special_tokens=True
    )

    # Controlla dove si trova
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id in tokens['input_ids'][0]:

        tokenizer_kwargs = {"truncation": True,  'max_length': max_lenght}
        results = mask_filler(masked_sentence, tokenizer_kwargs=tokenizer_kwargs)

        join = False
        for possible_sentence in results:

            if possible_sentence['sequence'] in joined_sentence:
                join = True
                break

        return join

    else:
        return False


def check_special_cases(list):

    if list[0][-1] == '(' or list[0][-1] == ')' or list[0][-1]=='’' or list[0][-1] == ',' or list[0][-1].isdigit() or list[0][-1] == '-'  or list[0][-1] =='>' or list[0][-1] =='<' or list[0][-1] ==':':
        return True
    else:
        return False

def check_sentence_similarity(main_sentence, other_sentences):

    cit = []
    no_cit = []
    # Confronta con ogni frase nelle altre_frasi
    for s in other_sentences:
        if s=="":
            continue

        citation = s['citation']
        text = s['text']

        splitted_citation = re.split(r'(?<=[\.·])\s*(?=[\.·]*\s*)', text) #re.split(r'(?<=[\.·])\s*', text)

        for s_cit in splitted_citation:
            if s_cit=="":
                continue

            if s_cit.strip() in main_sentence:
                cit.append({'citation': citation, 'text': s_cit})
            else:
                no_cit.append({'citation': citation, 'text': s_cit})

    return cit, no_cit


def split_phrases(phrases, all_citations, min_words_in_phrase, model_max_length, tokenizer):
    result = []
    result_cit = []

    current_citation_index = 0
    for idx, phrase in enumerate(phrases):
        # Dividiamo la frase basandoci sui delimitatori punto basso e punto alto
        parts = re.split(r'([\.·]+)', phrase)#re.split(r'(\.|\·)', phrase)  # Manteniamo i delimitatori

        # Ricostruzione delle sottofrasi mantenendo almeno N parole
        current_subphrase = ""

        if idx!=current_citation_index and current_citation_index<len(all_citations):
            for item in all_citations[current_citation_index]:
                all_citations[idx].append(item)
            current_citation_index = idx

        for i in range(0, len(parts) - 1, 2):
            subphrase = parts[i].strip()
            delimiter = parts[i + 1]  # Punto basso o punto alto

            if current_subphrase:
                subphrase = current_subphrase + " " + subphrase
                current_subphrase = ""

            word_count = len(subphrase.split())

            #check if next part is too small
            check = True
            if i < len(parts)-2 and (len(tokenizer(subphrase, return_tensors="pt")['input_ids'][0]) + len(tokenizer(parts[i+2], return_tensors="pt")['input_ids'][0]) < model_max_length):
                if (len(parts[i+2].strip().split()) < min_words_in_phrase or check_special_cases(parts[i+2].strip().split())):
                    check = False

            if word_count >= min_words_in_phrase and check:
                result.append(subphrase + delimiter)

                if current_citation_index < len(all_citations):
                    cit, no_cit = check_sentence_similarity(subphrase + delimiter, all_citations[current_citation_index])
                    result_cit.append(cit)
                else:
                    cit = []
                    no_cit = []

                if len(no_cit) == 0:
                    current_citation_index+=1
                else:
                    all_citations[current_citation_index] = no_cit
            else:
                current_subphrase = subphrase + delimiter

        # Aggiungiamo l'ultima sottofrase non ancora processata
        if current_subphrase:
            if result:
                result[-1] += " " + current_subphrase  # Accorpiamo all'ultima sottofrase
                if current_citation_index < len(all_citations):
                    cit, no_cit = check_sentence_similarity(current_subphrase, all_citations[current_citation_index])
                    result_cit[-1].append(cit)
                else:
                    result_cit.append([])
                    cit = []
                    no_cit = []

                if len(no_cit) == 0:
                    current_citation_index+=1
                else:
                    all_citations[current_citation_index] = no_cit
            else:
                result.append(current_subphrase)
                if current_citation_index < len(all_citations):
                    cit, no_cit = check_sentence_similarity(current_subphrase, all_citations[current_citation_index])
                    result_cit.append(cit)
                else:
                    result_cit.append([])
                    cit = []
                    no_cit = []

                if len(no_cit) == 0:
                    current_citation_index+=1
                else:
                    all_citations[current_citation_index] = no_cit

    return result, result_cit


def extract_dicts(nested_array):
    extracted = []

    # Funzione ricorsiva per navigare l'array
    def traverse(item):
        if isinstance(item, dict):  # Se è un dizionario
            extracted.append(item)
        elif isinstance(item, list):  # Se è una lista, esamina i suoi elementi
            for sub_item in item:
                traverse(sub_item)

    traverse(nested_array)
    return extracted


def extract_sentences_from_texts(
    texts, citations, mask_filler, min_words_in_phrase, model_max_length, tokenizer):

    sentences = []
    all_citations = []

    current_sentence = ""
    current_citation = []


    for i, text_i in enumerate(texts):

        #get current text
        current_text = text_i

        #get last meaningfull digit of the current text
        last_char = current_text.strip()[-1]

        #If the current sentence already has a Low dot (.) or a High dot (·), end the sentence
        if last_char == "." or last_char == "·":

            #if current sentence is empty
            if current_sentence == "":
                current_sentence = current_text

            else:

                # if current sentence is not empty, check if the first word of the current text
                # must be joined with the one at the end of the current sentence
                join = check_if_two_words_go_together(
                    current_sentence, current_text, mask_filler, model_max_length, tokenizer)

                #if must be joined
                if join:
                    current_sentence = merge_sentences_with_mask(
                        current_sentence, current_text, tokenizer, with_mask=False)
                else:
                    current_sentence = merge_sentences_with_mask(
                        current_sentence, current_text, tokenizer, with_mask=False, add_space_between_sentences=True)


            sentences.append(current_sentence)
            current_sentence = ""
            current_citation.append(citations[i])
            all_citations.append(current_citation)
            current_citation = []
            continue


        #otherwise
        else:

            #if current sentence is empty
            if current_sentence == "":
                current_sentence = current_text

            #if current sentence is not empty, check if the first word of the current text must be joined with the one at the end of the current sentence
            else:

                #predict if must be joined with the first word of the current text
                join = check_if_two_words_go_together(current_sentence, current_text, mask_filler, model_max_length, tokenizer)

                #if must be joined
                if join:
                    current_sentence = merge_sentences_with_mask(
                        current_sentence, current_text, tokenizer, with_mask=False)
                else:
                    current_sentence = merge_sentences_with_mask(
                        current_sentence, current_text, tokenizer, with_mask=False, add_space_between_sentences=True)


            current_citation.append(citations[i])

    if current_sentence != "":
        sentences.append(current_sentence)

    if len(current_citation) > 0:
        all_citations.append(current_citation)

    splitted_sentences, splitted_citations = split_phrases(sentences, all_citations, min_words_in_phrase, model_max_length, tokenizer)


    fixed_splitted_citations = []
    #fix citations
    for j in range(len(splitted_sentences)):
        if j<len(splitted_citations):
            fixed_splitted_citations.append(extract_dicts(splitted_citations[j]))
        else:
            fixed_splitted_citations.append(extract_dicts([]))


    return splitted_sentences, fixed_splitted_citations



def encode_sentences(sentences, model, tokenizer, len_embedding, device, model_max_lenght):

    encoding = np.zeros((len(sentences), len_embedding)).astype(np.float32)

    for i, sentence in enumerate(sentences):

        # Tokenize sentence
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=model_max_lenght).to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Use [CLS] token embedding as sentence encoding
        sentence_embedding = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy()

        # Normalize (lo faremo in faiss direttamente)
        #sentence_embedding = sentence_embedding / np.linalg.norm(sentence_embedding)

        encoding[i] = sentence_embedding

    return encoding