import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import (xavier_uniform_, constant_)

import pytorch_lightning as pl


class LinearBatchNorm(nn.Module):
    """Linear → BatchNorm1d → ReLU → Dropout building block."""

    def __init__(self, in_features: int, out_features: int, dropout: float, constant_size: bool, dim_batch: int | None = None):
        if dim_batch is None:
            dim_batch = out_features
        super(LinearBatchNorm, self).__init__()
        self.cs = constant_size
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            self.get_norm(dim_batch),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
    def get_norm(self, out_features: int) -> nn.BatchNorm1d:
        norm = nn.BatchNorm1d(out_features)
        return norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Implements the multihead attention mechanism used in 
    MultiHeadedAttentionMIL_*. 
    Input (batch, nb_tiles, features)
    Output (batch, nb_tiles, nheads)
    """
    def __init__(self, feature_depth: int, dropout: float = 0.1, num_heads: int = 1, atn_dim: int = 256):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.atn_layer_1_weights = nn.Parameter(torch.Tensor(atn_dim, feature_depth))
        self.atn_layer_2_weights = nn.Parameter(torch.Tensor(1, 1, self.num_heads, self.dim_heads, 1))
        self.atn_layer_1_bias = nn.Parameter(torch.empty((atn_dim)))
        self.atn_layer_2_bias = nn.Parameter(torch.empty((1, self.num_heads, 1, 1)))
        self._init_weights()

    def _init_weights(self) -> None:
        xavier_uniform_(self.atn_layer_1_weights)
        xavier_uniform_(self.atn_layer_2_weights)
        constant_(self.atn_layer_1_bias, 0)
        constant_(self.atn_layer_2_bias, 0)

    def forward(self, x):
        """ Extracts a series of attention scores.

        Args:
            x (torch.Tensor): size (batch, nb_tiles, features)

        Returns:
            torch.Tensor: size (batch, nb_tiles, nb_heads)
        """
        bs, nbt, _ = x.shape

        # Weights extraction
        x = F.linear(x, weight=self.atn_layer_1_weights, bias=self.atn_layer_1_bias)
        x = torch.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view((bs, nbt, self.num_heads, 1, self.dim_heads))
        x = torch.matmul(x , self.atn_layer_2_weights) + self.atn_layer_2_bias # 4 scores.
        x = x.view(bs, nbt, -1) # shape (bs, nbt, nheads) 
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, feature_depth: int = 768, num_tokens: int = 200, num_heads: int = 4, num_layers: int = 4, ff_dim: int = 2048, dropout: float = 0.1):
        super(SimpleTransformer, self).__init__()

        self.seq_length = num_tokens + 1  # +1 for CLS
        self.feature_depth = feature_depth

        # self.pos_embedding = Parameter(torch.randn(1, self.seq_length, feature_depth))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_depth, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            batch_first=True  # Required for batch (B, L, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token learnable
        self.cls_token = Parameter(torch.randn(1, 1, feature_depth))

    def forward(self, x):
        """
        x : Tensor (batch_size, num_tokens, feature_depth)
        """
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, feature_depth)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_tokens+1, feature_depth)
        # x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer_encoder(x)  # (B, num_tokens+1, feature_depth)

        return x

class PoolingFunction(nn.Module):
    """Aggregates tile embeddings into a slide representation.

    Supported pooling strategies: 'attmil' (multi-head attention MIL), 'mean', 'max'.
    """

    def __init__(
            self,
            feature_depth: int,
            pooling_fct: str = 'attmil',
            dropout: float = 0.1,
            num_heads: int = 1,
            atn_dim: int = 256,
        ):
        super(PoolingFunction, self).__init__()
        self.pooling = pooling_fct
        if self.pooling == 'attmil' or self.pooling == 'inst_attmil':
            self.attention = nn.Sequential(
                MultiHeadAttention(feature_depth, dropout=dropout, num_heads=num_heads, atn_dim=atn_dim),
                nn.Softmax(dim=-2)
            )

    def forward(self, x: torch.Tensor, scores: torch.Tensor | None = None) -> torch.Tensor:
        """Pool tile embeddings (or pre-computed scores) into a slide representation.

        Args:
            x: Tile embeddings of shape (B, N, F).
            scores: Optional pre-computed instance scores of shape (B, N, C),
                    used only when pooling == 'attmil'.
        """
        if self.pooling == 'attmil':
            w = self.attention(x) # (bs, nbt, nheads)
            w = torch.transpose(w, -1, -2) # (bs, nheads, nbt)
            if scores is None:
                slide = torch.matmul(w, x) # Slide representation, shape (bs, nheads, nfeatures)
            else:
                slide = torch.matmul(w, scores) # Slide representation, shape (bs, nheads, num_classes)
            slide = slide.flatten(1, -1) # (bs, nheads*nfeatures)

        elif self.pooling == 'mean':
            slide = torch.mean(x, -2) # BxF
        
        elif self.pooling == 'max':
            slide = torch.max(x, -2) # BxF

        else:
            print(f'{self.pooling} pooling function not yet implemented``')
        return slide

class BulkMIL(pl.LightningModule):
    """
    General MIL algorithm with tunable pooling function.
    Same as MultiHeadedAttentionMIL_multiclass but have a classifier with N 
    Linear layers. 
    """
    def __init__(
            self, 
            feature_depth, 
            num_genes,
            num_classes,
            pooling_fct = "attmil",
            instance_based = False,
            classif_on_gene_exp = False,
            dropout = 0.1, 
            num_heads = 1, 
            atn_dim = 256,
            predictor_hidden_dims = [],
            classifier_hidden_dims = [],
            encoder_type = "identity",
            nb_tiles = None,
            encoder_num_heads=4,
            encoder_num_layers=1, 
            encoder_ff_dim=2048, 
            encoder_dropout=0.3,
        ):
        super(BulkMIL, self).__init__()

        ##  Set parameters
        self.dropout = dropout
        self.pooling_fct = pooling_fct
        self.instance_based = instance_based
        self.classif_on_gene_exp = classif_on_gene_exp
        self.feature_depth = feature_depth
        self.predictor_hidden_dims = predictor_hidden_dims
        self.classifier_hidden_dims = classifier_hidden_dims
        self.num_heads = num_heads
        self.num_genes = num_genes
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.nb_tiles = nb_tiles
        self.encoder_type = encoder_type
        self.num_classes = num_classes
        if encoder_type=="vit":
            assert self.nb_tiles is not None, "nb_tiles is needed for ViT encoder."
        self.encoder_num_heads = encoder_num_heads
        self.encoder_num_layers = encoder_num_layers
        self.encoder_ff_dim = encoder_ff_dim
        self.encoder_dropout = encoder_dropout

        ## set networks
        self.encoder = self._build_encoder()
        self.pooling_function = PoolingFunction(
            feature_depth, 
            pooling_fct=pooling_fct, 
            dropout=dropout, 
            num_heads=num_heads, 
            atn_dim=atn_dim
        )
        if self.num_genes!=0:
            self.predictor = self._build_predictor()
        self.classifier = self._build_classifier()

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_pooling(self) -> None:
        for param in self.pooling_function.parameters():
            param.requires_grad = False

    def freeze_predictor(self) -> None:
        for param in self.predictor.parameters():
            param.requires_grad = False

    def freeze_classifier(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = False

    def _build_encoder(self) -> nn.Module:
        """Build the tile encoder: Identity or SimpleTransformer (ViT)."""
        if self.encoder_type=="identity":
            return nn.Identity()
        elif self.encoder_type=="vit":
            encoder = SimpleTransformer(
                feature_depth = self.feature_depth,
                num_tokens=self.nb_tiles, 
                num_heads=self.encoder_num_heads, 
                num_layers=self.encoder_num_layers, 
                ff_dim=self.encoder_ff_dim, 
                dropout=self.encoder_dropout,
            )
            return encoder
        else:
            NotImplementedError
        return 

    def _build_predictor(self) -> nn.Sequential:
        """Build the gene-expression predictor MLP (hidden dims → num_genes)."""
        predictor = []
        prev_dim = int(self.feature_depth * self.num_heads)
        for hidden_dim in self.predictor_hidden_dims:
            predictor.append(
                LinearBatchNorm(prev_dim, hidden_dim, self.dropout, True))
            prev_dim = hidden_dim
        predictor.append(nn.Linear(self.predictor_hidden_dims[-1], self.num_genes))
        return nn.Sequential(*predictor)
        
    def _build_classifier(self) -> nn.Sequential:
        """Build the subtype classifier MLP (hidden dims → num_classes)."""
        classifier = []
        prev_dim = int(self.feature_depth)
        if self.classif_on_gene_exp:
            prev_dim = self.num_genes
        for hidden_dim in self.classifier_hidden_dims:
            classifier.append(
                LinearBatchNorm(prev_dim, hidden_dim, self.dropout, True))
            prev_dim = hidden_dim
        classifier.append(nn.Linear(self.classifier_hidden_dims[-1], self.num_classes))
        return nn.Sequential(*classifier)
        
    
    def forward(self, x):
        """
    Input x of size BxNxF where :
            * F is the dimension of feature space
            * N is number of patche
    """
        bs, nbt, emb_size = x.shape
        x = self.encoder(x)
        if self.encoder_type == "vit":
            x_classif, x_genes = x[:, 0, :], x[:, 1:, :].contiguous()
        else:
            x_classif, x_genes = x, x 
        
        if  self.instance_based: # Instance Based MIL
            gene_exp, classif = self.instance_based_mil(
                x_genes, x_classif, emb_size, bs, nbt)

        else: # Feature Based MIL
            gene_exp, classif = self.feature_based_mil(
                x_genes, x_classif, emb_size, bs, nbt)

        return gene_exp, classif

    def feature_based_mil(self, x_genes: torch.Tensor, x_classif: torch.Tensor, emb_size: int, bs: int, nbt: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Slide-level MIL: pool tile embeddings first, then predict gene exp and class."""
        x_genes = self.pooling_function(x_genes)
        if self.encoder_type != "vit":
            x_classif = x_genes

        gene_exp = self.predictor(x_genes)
        if self.classif_on_gene_exp:
            classif = self.classifier(gene_exp)
        else:
            classif = self.classifier(x_classif)
        return gene_exp, classif
    
    def instance_based_mil(self, x_genes: torch.Tensor, x_classif: torch.Tensor, emb_size: int, bs: int, nbt: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Instance-level MIL: predict per-tile gene exp and class, then pool."""
        # Gene expression prediction
        gene_exp = self.predictor(x_genes.view(-1, emb_size)) # bs x nbt, emb_size
        gene_exp = gene_exp.view(bs, nbt, self.num_genes) # bs, nbt, num_genes
        if self.pooling_fct=="attmil":
            gene_exp = self.pooling_function(x_genes, gene_exp) # bs, num_genes
        else:
            gene_exp = self.pooling_function(gene_exp) # bs, num_genes

        # Classification
        if self.classif_on_gene_exp:
            classif = self.classifier(gene_exp)
        else:
            if self.encoder_type == "vit":
                classif = self.classifier(x_classif) # bs, num_classes
            else:
                classif = self.classifier(x_classif.view(-1, emb_size)) # bs x nbt, num_classes
                classif = classif.view(bs, nbt, self.num_classes) # bs, nbt, num_classes
                if self.pooling_fct=="attmil":
                    classif = self.pooling_function(x_classif, classif)
                else:
                    classif = self.pooling_function(classif)

        return gene_exp, classif
    
    def forward_per_tile(self, x):
        """
        Input x of size BxNxF where :
                * F is the dimension of feature space
                * N is number of patche
        """
        bs, nbt, emb_size = x.shape
        x = self.encoder(x)
        if self.encoder_type == "vit":
            x_genes = x[:, 1:, :].contiguous()
            return "Not Implemented"

        gene_exp = self.predictor(x.view(-1, emb_size)) # bs x nbt, emb_size
        gene_exp = gene_exp.view(bs, nbt, self.num_genes) # bs, nbt, num_genes

        if self.classif_on_gene_exp:
            classif = self.classifier(gene_exp.view(-1, self.num_genes))
            classif = classif.view(bs, nbt, self.num_classes)
        else:
            classif = self.classifier(x.view(-1, emb_size))
            classif = classif.view(bs, nbt, self.num_classes)

        return gene_exp, classif
    
