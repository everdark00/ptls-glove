from collections import OrderedDict

import torch
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from .glove_embedding import GloveEmbedding
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase


class TrxEncoderGlove(nn.Module):
    def __init__(self,  
                 glove_embedding : GloveEmbedding = None,
                 agg_type="cat"
                 ):
        super().__init__()

        self.agg_type = agg_type
        self.feature_names = glove_embedding.feature_names
        self.embedding_vectors = glove_embedding.get_vectors(agg_type="mean")
                
    

    def forward(self, x: PaddedBatch):
        if self.agg_type == "cat":
            out = []
            for fe in self.feature_names:
                out.append(self.embedding_vectors(x.payload[fe]))
            out = torch.cat(out, dim=2)
            return PaddedBatch(out, x.seq_lens)
        else:
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
            return self.embedding_vectors.weight.shape[1] * len(self.feature_names)
        else:
            return self.embedding_vectors.weight.shape[1]

class TrxEncoderCat(TrxEncoderBase):
    def __init__(self,  
                 embeddings,
                 embeddings_noise=0.003,
                 emb_dropout=0,
                 spatial_dropout=False,
                 agg_type: str = "cat",
                 out_of_index: str = 'clip',
                 ):
        super().__init__()

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

        self.agg_type = agg_type    

    def forward(self, x: PaddedBatch):
        processed_embeddings = []

        for field_name in self.embeddings.keys():
            processed_embeddings.append(self.get_category_embeddings(x, field_name))

        if self.agg_type == "cat":
            out = torch.cat(processed_embeddings, dim=2)
        else:
            n_emb = 0
            out = None
            for i, emb in enumerate(processed_embeddings):
                out = emb if i == 0 else out + emb
                n_emb += 1
            if self.agg_type == "mean":
                out = out / n_emb

        return PaddedBatch(out, x.seq_lens)


    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        for e in self.embeddings.values():
            esz = e.embedding_dim
            break
        if self.agg_type == "cat":
            return esz * len(self.embeddings)
        else:
            return esz