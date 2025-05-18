from collections import OrderedDict

import torch
from torch import nn as nn
import pandas as pd
import numpy as np

from pdb import set_trace

from ptls.data_load.padded_batch import PaddedBatch
from .glove_embedding import GloveEmbedding, TransEmbedding
from .time2vec import Time2VecModule
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase


class TrxEncoderGlove(nn.Module):
    def __init__(self,  
                 glove_embedding : GloveEmbedding = None,
                 agg_type="cat",
                 numeric_separate = False,
                 numeric_features=[]
                 ):
        super().__init__()
        self.numeric_separate = numeric_separate
        self.numeric_features = numeric_features
        self.agg_type = agg_type
        self.feature_names = glove_embedding.feature_names
        self.embedding_vectors = glove_embedding.get_vectors(agg_type="mean")
                
    def forward(self, x: PaddedBatch):
        if self.agg_type == "cat":
            out = []
            for fe in self.feature_names:
                out.append(self.embedding_vectors(x.payload[fe]))
            if self.numeric_separate:
                for fe in self.numeric_features:
                    out.append(x.payload[fe])
            out = torch.cat(out, dim=2)
            return PaddedBatch(out, x.seq_lens)
        else:
            if self.numeric_separate:
                raise Exception("Only cat agg alowed with numeric separate")
            out = self.embedding_vectors(x.payload[self.feature_names[0]])
            for fe in self.feature_names[1:]:
                out += self.embedding_vectors(x.payload[fe])
            if self.agg_type == "sum":
                return PaddedBatch(out, x.seq_lens)
            else:
                return PaddedBatch(out/len(self.feature_names), x.seq_lens)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.agg_type == "cat":
            return self.embedding_vectors.weight.shape[1] * len(self.feature_names)  + (0 if not self.numeric_separate else len(self.numeric_features))
        else:
            return self.embedding_vectors.weight.shape[1]

class TrxEncoderCat(TrxEncoderBase):
    def __init__(self,  
                 embeddings,
                 id_col_name,
                 numeric_separate=False,
                 numeric_id=False,
                 numeric_features=None,
                 time_features=None,
                 time_proj_method='plain',
                 time2vec_hs=0,
                 text_embeddings_path=None,
                 text_embedding_proj=False,
                 text_embeddings_sz=0,
                 embeddings_noise=0.003,
                 emb_dropout=0,
                 spatial_dropout=False,
                 agg_type: str = "cat",
                 out_of_index: str = 'clip',
                 ):
        self.numeric_separate = numeric_separate
        self.numeric_features = numeric_features
        self.numeric_id = numeric_id

        self.text_embeddings_path = text_embeddings_path
        self.text_esz = text_embeddings_sz
        self.text_embedding_proj = text_embedding_proj

        self.time_features=time_features
        self.time_proj_method = time_proj_method
        self.time2vec_hs=time2vec_hs

        self.id_col_name = id_col_name
        self.device = 'cuda'
        self.esz = None

        if self.numeric_separate and self.numeric_id:
            raise Exception(f'"numeric_id" and "numeric_separate" could not be applied together!')

        noisy_embeddings = {}
        for emb_name, emb_props in embeddings.items():
            if emb_props.get('disabled', False):
                continue
            if emb_props['in'] == 0 or emb_props['out'] == 0:
                continue
            noisy_embeddings[emb_name] = NoisyEmbedding(
                num_embeddings=emb_props['in'],
                embedding_dim=emb_props['out'],
                padding_idx=0,
                max_norm=None,
                noise_scale=embeddings_noise,
                dropout=emb_dropout,
                spatial_dropout=spatial_dropout,
            )

        super().__init__(
            embeddings=noisy_embeddings,
            numeric_values=None,
            custom_embeddings={},
            out_of_index=out_of_index,
        )

        if not self.numeric_id:
            for e in self.embeddings.values():
                self.esz = e.embedding_dim
                break
        else:
            self.esz = 0
            for e in self.embeddings.values():
                self.esz += e.embedding_dim

        self.text_embeddings = dict()
        self.text_proj_module = nn.ModuleDict()
        if self.text_embeddings_path is not None:
            text_embeddings_vectors = torch.load(self.text_embeddings_path, weights_only=True)
            for fn in text_embeddings_vectors.keys():
                self.text_embeddings[fn] = nn.Embedding(num_embeddings=text_embeddings_vectors[fn].shape[0], embedding_dim=self.text_esz)
                self.text_embeddings[fn].weight = nn.Parameter(text_embeddings_vectors[fn])
                self.text_embeddings[fn].weight.requires_grad = False
                self.text_embeddings[fn].to(self.device)

            if self.text_embedding_proj:
                for fn in text_embeddings_vectors.keys():
                    self.text_proj_module[fn] = nn.Sequential(
                            nn.Linear(self.text_esz, self.esz),
                            nn.ReLU()
                        ).to(self.device)
                self.text_esz = self.esz
            else:
                if agg_type != 'cat' and self.text_esz != self.esz:
                    raise Exception(f'General rep size is {self.esz} and text embedding size is {self.text_esz}. Add proj layer or change embedding dimensionality!')
            
            del text_embeddings_vectors

        self.time_proj_module = nn.ModuleDict()
        self.time_out_size=self.esz
        if len(time_features) == 0:
            self.time_out_size=0
        if self.time_proj_method != 'plain':
            if len(time_features) == 0:
                raise Exception('ERROR t2v time encoding initiated, but passed list of time features is empty')
            for fn in self.time_features:
                self.time_proj_module[fn] = Time2VecModule('sin', self.time2vec_hs).to(self.device)

#            if not self.numeric_id:
            self.time_out_size =  self.esz if (not self.numeric_id) else (self.esz // len(self.embeddings.keys()))
            self.time_proj_module['proj'] = nn.Sequential(
                            nn.Linear(self.time2vec_hs * len(self.time_features), self.time_out_size),
                            nn.ReLU()
                        ).to(self.device)


        self.agg_type = agg_type  

    def forward(self, x: PaddedBatch):
        processed_embeddings = []

        if self.numeric_id:
            for fn in self.numeric_features:
                processed_embeddings.append(x.payload[fn].unsqueeze(2))

        if self.numeric_separate:
            for fn in self.numeric_features:
                value = x.payload[f'{fn}_val']
                pos = x.payload[f'{fn}_pos'] 
                numeric_embedding = torch.ones((pos.shape[0], pos.shape[1], self.esz)).double().to(self.device)
                zero_mask = torch.ones(pos.shape[0], pos.shape[1]).double().to(self.device)
                for i in range(self.esz):
                    if i > 0:
                        numeric_embedding[:, :, i] *= zero_mask
                    numeric_embedding[:, :, i][pos == i + 1] = value[pos == i + 1]
                    numeric_embedding[:, :, i][pos == 0] = 0
                    zero_mask *= (pos != i + 1)
                processed_embeddings.append(numeric_embedding)

        if self.text_embeddings_path is not None:
            for fn in self.text_embeddings.keys():
                if self.text_embedding_proj:
                    processed_embeddings.append(self.text_proj_module[fn](self.text_embeddings[fn](x.payload['text_emb_id'])))
                else:
                    processed_embeddings.append(self.text_embeddings[fn](x.payload['text_emb_id']))

        if self.time_proj_method != 'plain':
            time_vectors = []
            for fn in self.time_features:
                time_vectors.append(self.time_proj_module[fn](x.payload[fn].view(-1, 1).float()))
            #if not self.numeric_id:
            processed_embeddings.append(self.time_proj_module['proj'](torch.cat(time_vectors, dim=1)).view(-1, x.payload[fn].shape[1], self.time_out_size))
            #else:
            #    processed_embeddings.append((torch.cat(time_vectors, dim=1)).view(-1, x.payload[fn].shape[1], self.time2vec_hs * len(self.time_features)))

        for field_name in self.embeddings.keys():
            processed_embeddings.append(self.get_category_embeddings(x, field_name))

        if self.agg_type == "cat":
            out = torch.cat(processed_embeddings, dim=2)
        else:
            if self.numeric_id:
                raise Exception(f'"numeric_id" param is True, numeric features are not discretized and embedded, only CAT agg allowed')
            n_emb = 0
            out = None
            for i, emb in enumerate(processed_embeddings):
                out = emb if i == 0 else out + emb
                n_emb += 1
            if self.agg_type == "mean":
                out = out / n_emb
        
        return PaddedBatch(out.float(), x.seq_lens)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.agg_type == "cat":
            if self.numeric_id is None:
                return self.esz * (len(self.embeddings) + (len(self.numeric_features) if self.numeric_separate else 0)) + self.text_esz * len(self.text_embeddings.keys()) + self.time_out_size
            else:
                return self.esz + len(self.numeric_features) + self.text_esz * len(self.text_embeddings.keys()) + self.time_out_size
        else:
            return self.esz

'''
algos

orig: obtain embeddings of size len(feature_names), sum them and sum with raw features 
classic: num discr + different aggregations on obtained embeds
'''
class TrxEncoderTrans(nn.Module):
    def __init__(self,  
                 feature_names, #must be cat or discretized num
                 algo='orig',
                 in_emb_sizes=[],
                 out_emb_size=None,
                 agg_type="cat",
                 numeric_separate=False,
                 numeric_features=[],
                 ):
        super().__init__()

        if algo not in {'orig', 'classic'}:
            raise Exception('algo must be "orig" or "classic"')

        self.numeric_separate = numeric_separate
        self.numeric_features = numeric_features

        self.device = 'cpu'
        self.algo = algo
        if self.algo == 'orig':
            self.esz = len(feature_names) + len(numeric_features)
        else:
            self.esz = out_emb_size

        self.agg_type = agg_type
        
        self.feature_names = feature_names
        self.embeddings = TransEmbedding(feature_names, in_emb_sizes, self.esz, self.device, 0.2, self.algo)
                
    

    def forward(self, x: PaddedBatch):
        if self.algo == 'orig':
            out = self.embeddings(x)

            out += torch.cat([x.payload[i].unsqueeze(2) for i in self.numeric_features + self.feature_names], dim=2)
        else:
            if self.agg_type == "cat":
                out = self.embeddings(x)
                for fe in self.numeric_features:
                    out.append(x.payload[fe].unsqueeze(2))
                out = torch.cat(out, dim=2)
                return PaddedBatch(out, x.seq_lens)
            else:
                if self.numeric_separate:
                    raise Exception('mean and sum agg does not supports using non-disc numeric features')
                out = torch.sum(self.embeddings(x), dim=2)
                
                if self.agg_type == "mean":
                    out = out / len(self.feature_names)
                    
        return PaddedBatch(out, x.seq_lens)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.algo == 'orig':
            return self.esz
        else:
            if self.agg_type == "cat":
                return self.esz * (len(self.embeddings.features)) + (len(self.numeric_features) if self.numeric_separate else self.esz) 
            else:
                return self.esz