from sklearn.decomposition import PCA

from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, util


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

        self.unk_str = None

    def calc_embeddings(self, X, idcol_name, fit_pca=True):
        ids = []
        for sample in X:
            ids.append(sample[idcol_name])
        
        embeds = dict()
        for fe in self.feature_names:
            def vclear(s):
                s = " ".join(s[fe])
                return re.sub(r"[^а-яА-Я0-9\s]+|\s{2,}", " ", s.lower()).strip().split(" ")

            if self.method == 'rubert_output':

                def get_emb(sample):
                    return self.model.encode(sample[fe])

                embeds[fe] = list(map(get_emb, X))
            elif self.method == 'avg_pooling_w2v':
                if self.unk_str is None:
                    vocabulary = list(set(np.concatenate(list(map(vclear, X)))) - {"", " "})
                    self.unk_str = self.unk_detection(vocabulary)

                def sclear(s):
                    sts = []
                    for st in s[fe]:
                        st = re.sub(r"[^а-яА-Я0-9\s]+|\s{2,}", " ", st.lower())
                        if self.unk_str != '':
                            st = re.sub(rf"{self.unk_str}", "unk", st)
                        sts.append(np.mean(self.model[*re.sub(r"\s+", " ", st).strip().split(" ")], axis=0))
                    return sts

                embeds[fe] = list(map(sclear, X))

            elif self.method == 'avg_pooling_fasttext':
                if self.unk_str is None:
                    vocabulary = list(set(np.concatenate(list(map(vclear, X)))) - {"", " "})
                    self.unk_str = self.unk_detection(vocabulary)

                def sclear(s):
                    sts = []
                    for st in s[fe]:
                        st = re.sub(r"[^а-яА-Я0-9\s]+|\s{2,}", " ", st.lower())
                        if self.unk_str != '':
                            st = re.sub(rf"{self.unk_str}", "unk", st)
                        sts.append(self.model.get_sentence_vector(re.sub(r"\s+", " ", st).strip()))
                    return sts

                embeds[fe] = list(map(sclear, X))

            if self.enable_pca:
                if fit_pca:
                    self.pca_models[fe].fit(np.concatenate(embeds[fe]))
                embeds[fe] = list(map(self.pca_models[fe].transform, embeds[fe]))

        return pd.DataFrame({idcol_name : ids} | embeds)

    def unk_detection(self, voc):
        unk_str = ""
        for w in voc:
            try:
                self.model[w]
            except:
                unk_str = unk_str + w + "|"
        return unk_str[:-1]