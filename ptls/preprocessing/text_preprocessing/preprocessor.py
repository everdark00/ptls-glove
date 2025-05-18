from sklearn.decomposition import PCA

from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, util
from tqdm.notebook import tqdm
import fasttext
import re
import torch
from pdb import set_trace

import pandas as pd
import numpy as np


class TextPreprocessor:
    def __init__(self, 
                 method, 
                 feature_names, 
                 enable_pca,  
                 compressed_dim=-1, 
                 saves_directory=None
                 ):
        self.compressed_dim = compressed_dim
        self.feature_names = feature_names
        self.enable_pca = enable_pca
        self.method = method
        if self.method == 'rubert_output':
            self.model = SentenceTransformer('sergeyzh/rubert-tiny-turbo')
            self.embedding_dim = 312
        elif self.method == 'avg_pooling_w2v':
            self.model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_ruwiki_20180420_300d", filename="ruwiki_20180420_300d.txt"))
            self.embedding_dim = 300
        elif self.method == 'avg_pooling_fasttext':
            self.model = fasttext.load_model(hf_hub_download(repo_id="facebook/fasttext-ru-vectors", filename="model.bin")) 
            self.embedding_dim = 300
        else:
            raise Exception('No method of text feature encoding with such name exists')

        self.pca_models = dict()
        if self.enable_pca:
            for fe in self.feature_names:
                self.pca_models[fe] =  PCA(n_components=self.compressed_dim)

        self.unk_str = dict()
        self.embed_idx = 1

    def calc_embeddings(self, X, fit_pca=True):   
        embeds = dict()
        for fe in self.feature_names:
            zero_embed = []
            sents = X[fe].values
            def vclear(s):
                s = " ".join(s)
                return re.sub(r"[^а-яА-Я0-9\s]+|\s{2,}", " ", s.lower()).strip().split(" ")

            if self.method == 'rubert_output':
                batch_size = 10000
                embeds_list = []

                for i in range(0, (X.shape[0] // batch_size) + 1):
                    if i * batch_size == X.shape[0]:
                        break
                    embeds_list.append(self.model.encode(sents[i * batch_size : min((i + 1) * batch_size, X.shape[0])]))

                if self.embed_idx  == 1:
                    zero_embed = list(self.model.encode([""]))

                sents = zero_embed + list(np.concatenate(embeds_list))

            elif self.method == 'avg_pooling_w2v':
                if fe not in self.unk_str:
                    vocabulary = list(set(vclear(sents)) - {"", " "})
                    self.unk_str[fe] = self.unk_detection(vocabulary)

                def sclear(s):
                    s = re.sub(r"[^а-яА-Я0-9\s]+|\s{2,}", " ", s.lower())
                    if self.unk_str[fe] != '':
                        s = re.sub(rf"{self.unk_str[fe]}", "unk", s)
                    return np.mean(self.model[*re.sub(r"\s+", " ", s).strip().split(" ")], axis=0)

                if self.embed_idx  == 1:
                    zero_embed = [self.model['unk']] 

                sents = zero_embed + list(map(sclear, sents))

            elif self.method == 'avg_pooling_fasttext':
                if fe not in self.unk_str:
                    vocabulary = list(set(vclear(sents)) - {"", " "})
                    self.unk_str[fe] = self.unk_detection(vocabulary)

                def sclear(s):
                    s = re.sub(r"[^а-яА-Я0-9\s]+|\s{2,}", " ", s.lower())
                    if self.unk_str[fe] != '':
                        s = re.sub(rf"{self.unk_str[fe]}", "unk", s)
                    return self.model.get_sentence_vector(re.sub(r"\s+", " ", s).strip())

                if self.embed_idx == 1:
                    zero_embed = [self.model.get_sentence_vector("")]

                sents = zero_embed + list(map(sclear, sents))

            if self.enable_pca:
                if fit_pca:
                    self.pca_models[fe].fit(sents)
                embeds[fe] = torch.tensor(self.pca_models[fe].transform(sents))
            else:
                embeds[fe] = torch.tensor(np.array(sents))

        self.embed_idx += X.shape[0]

        return embeds

    def unk_detection(self, voc):
        unk_str = ""
        for w in voc:
            try:
                self.model[w]
            except:
                unk_str = unk_str + w + "|"
        return unk_str[:-1]