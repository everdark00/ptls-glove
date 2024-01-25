from collections import OrderedDict

import torch
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from .glove_embedding import GloveEmbedding


class TrxEncoderGlove(nn.Module):
    def __init__(self,  
                 glove_embedding : GloveEmbedding =   None,
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