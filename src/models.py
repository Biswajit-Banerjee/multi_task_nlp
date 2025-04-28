import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pos and i (dimension)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CustomEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        # token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # one layer of the Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        # stack N such layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # initialize parameters following the original paper
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of token indices, shape (seq_len, batch_size)
            attention_mask: optional mask of shape (seq_len, seq_len)
        Returns:
            output: (seq_len, batch_size, d_model)
        """
        if attention_mask is not None:
            mask = (attention_mask == 0)
        else:
            mask = None

        # embed tokens and scale
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        # add positional encoding
        x = self.pos_encoder(x)

        # swap to (T, B, D)
        x = x.transpose(0, 1)

        # pass through the stack of encoder layers
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        # (B, T, D)
        output = output.transpose(0, 1)

        # mimic output compatibility with HF
        class BackboneOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return BackboneOutput(last_hidden_state=output)


class SentenceEncoder(nn.Module):
    """
    Wraps a transformer backbone and mean-pooling layer to produce sentence embeddings.
    """

    def __init__(self, backbone_name: str = "bert-base-uncased", embedding_dim=512):
        super().__init__()

        if backbone_name != "custom":
            self.backbone = AutoModel.from_pretrained(backbone_name)
        else:
            self.backbone = CustomEncoder(d_model=embedding_dim)

    def forward(self, input_ids, attention_mask):
        # Transformer outputs last_hidden_state of shape [B, T, H]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # Mean-pooling mask out padding tokens
        masked_sum = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True)
        return masked_sum / lengths


class ATISMultiTaskModel(nn.Module):
    """
    Joint model for intent classification (Task A) and slot filling (Task B).
    """

    def __init__(
        self,
        encoder: SentenceEncoder,
        n_intents: int = 8,
        n_slots: int = 10,
    ):
        super().__init__()
        self.encoder = encoder
        
        # extract d_model
        hidden_dim = encoder.backbone.embedding.embedding_dim 

        # projection for intent
        self.intent_head = nn.Linear(hidden_dim, n_intents)
        
        # projection for entities/slots
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2,
            activation='relu'
        )
        projection = nn.Linear(hidden_dim, n_slots)
        
        self.slot_head = nn.Sequential(
            encoder_layer, 
            projection
        )

    def forward(self, input_ids, attention_mask):
        # Sentence-level embedding for intent
        sent_emb = self.encoder(input_ids, attention_mask)  # [B, H]
        logits_intent = self.intent_head(sent_emb)  # [B, n_intents]

        # Token-level representations for entities
        token_outputs = self.encoder.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [B, T, H]
        logits_slots = self.slot_head(token_outputs)  # [B, T, n_slots]
        return logits_intent, logits_slots
