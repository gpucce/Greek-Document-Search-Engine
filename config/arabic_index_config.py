# pylint: disable=pointless-string-statement
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()


    config.run = run = ConfigDict()

    #gpu device
    run.device = 2

    config.data = data = ConfigDict()

    '''
        The dataset is in this format:
            - dataset_name/
                - year block/data/
                    - dataset1/
                        - dataset1.1/
                            - <...>.ara1 or <...>.ara1.yml
                        - dataset1.2/
                            - <...>.ara1 or <...>.ara1.yml
                    - dataset2/
                        - dataset2.1/
                            - <...>.ara1 or <...>.ara1.yml
                ...

        So, json_dataset_path is the absolute path to 'dataset_name'
    '''
    data.json_dataset_path = "/home/gpucce/Repos/arabo_panzeca/all_data/"
    data.date_range = [600, 800]

    '''
    When creating the index, sentences are broken up into low and high points. However, some sentences, if broken up, do not make much sense on their own.

    e.g. ...τῆς βασιλείας αὐτοῦ νῦν καὶ εἰς τοὺς αἰῶνας τῶν αἰώνων. ἀμήν.

    It is a prayer that ends with "amen". It would be ideal to keep it in the sentence.
    The min_words_in_phrase parameter indicates how many words at least a sentence must have to be "alone", separate from the others.
    '''
    data.min_words_in_phrase = 5

    #lenght of the embedding of each sentence. Will be used to inizialize the index
    data.len_embedding = 768

    #dove si trova la lista contenente le opere TLG (create prima con lo script save_name_of_the_works)
    data.name_of_the_works_path = "/home/gpucce/Greek-Document-Serach-Engine/name_of_openiti_works.npy"

    config.model = model = ConfigDict()

    model.tokenizer = "aubmindlab/bert-base-arabertv2"
    model.model = "aubmindlab/bert-base-arabertv2"
    assert model.model == model.tokenizer, "can't use different tokenizer and model"
    if model.model == "aubmindlab/bert-large-arabertv2":
        assert data.len_embedding == 1024, "len_embedding should be 1024 if using bert-large-arabertv2"
    elif model.model == "aubmindlab/bert-base-arabertv2":
        assert data.len_embedding == 768, "len_embedding should be 768 if using bert-base-arabertv2"

    #max number of tokens the model can handle
    model.model_max_length = 512
    #when decide to join two words w1-w2, check if w2 is in the top_k next words after w1
    model.top_k = 10

    config.index = index = ConfigDict()
    index.index_path = "/home/gpucce/arabo_panzeca/"
    index.index_name = "Faiss_Arabic"

    config.db = db = ConfigDict()
    db.db_path = "/home/gpucce/arabo_panzeca/"
    db.db_name = "DB_Arabic"

    #not important. Just if you want to execute test_index.py
    config.retrieval = retrieval = ConfigDict()
    retrieval.num_matches = 100

    return config
