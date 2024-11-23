"""Advanced LLM-based document classification model implementation.

This module implements a sophisticated document classification model using a custom
transformer-based architecture. It uses pre-trained LLM weights as initialization
and adds multiple transformer layers with advanced attention mechanisms.

The model architecture consists of:
    1. Pre-trained LLM base model (DeBERTa-v3)
    2. Multiple transformer encoder layers with:
        - Multi-head self-attention with relative position encoding
        - Position-wise feed-forward networks
        - Layer normalization and residual connections
    3. Advanced pooling mechanisms
    4. Multiple classification heads with intermediate supervision

Typical usage:
    ```python
    model = AdvancedDocClassificationModel(
        n_doc_labels=10,
        p_dropout=0.1,
        num_hidden_layers=2
    )
    outputs = model(input_ids, attention_mask)
    logits = outputs["logits"]
    ```
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative position encoding.

    This module implements multi-head attention with the following features:
        1. Scaled dot-product attention
        2. Relative position encoding using learned embeddings
        3. Residual connection and layer normalization
        4. Attention dropout for regularization

    Attributes:
        num_attention_heads (int): Number of attention heads
        attention_head_size (int): Size of each attention head
        all_head_size (int): Total size of all attention heads
        query (nn.Linear): Query transformation
        key (nn.Linear): Key transformation
        value (nn.Linear): Value transformation
        dropout (nn.Dropout): Dropout layer
        dense (nn.Linear): Output projection
        LayerNorm (nn.LayerNorm): Layer normalization
        max_position_embeddings (int): Maximum sequence length
        distance_embedding (nn.Embedding): Relative position embeddings
    """

    def __init__(self, config):
        """Initialize the multi-head attention module.

        Args:
            config: Configuration object containing:
                - num_attention_heads: Number of attention heads
                - hidden_size: Size of hidden states
                - attention_probs_dropout_prob: Dropout probability
                - max_position_embeddings: Maximum sequence length
                - layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Relative position encoding
        self.max_position_embeddings = config.max_position_embeddings
        self.distance_embedding = nn.Embedding(
            2 * config.max_position_embeddings - 1, self.attention_head_size
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and reshape tensor for attention computation.

        Args:
            x: Input tensor of shape (batch_size, seq_length, all_head_size)

        Returns:
            torch.Tensor: Reshaped tensor of shape
                (batch_size, num_heads, seq_length, head_size)
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention with relative position encoding.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional mask tensor of shape (batch_size, seq_length)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: Attended tensor of shape (batch_size, seq_length, hidden_size)
                - attention_probs: Attention probabilities of shape
                    (batch_size, num_heads, seq_length, seq_length)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Calculate relative position encodings
        position_ids = torch.arange(
            hidden_states.size(1), dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(hidden_states[:, :, 0])

        relative_positions = position_ids.unsqueeze(2) - position_ids.unsqueeze(1)
        relative_positions += self.max_position_embeddings - 1
        relative_position_scores = self.distance_embedding(relative_positions)
        relative_position_scores = relative_position_scores.permute(0, 3, 1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + relative_position_scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # Add & Norm
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        return attention_output, attention_probs


class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    This module implements a two-layer feed-forward network with:
        1. GELU activation between layers
        2. Residual connection
        3. Layer normalization
        4. Dropout for regularization

    Attributes:
        dense1 (nn.Linear): First linear transformation
        intermediate_act_fn (nn.GELU): GELU activation
        dense2 (nn.Linear): Second linear transformation
        LayerNorm (nn.LayerNorm): Layer normalization
        dropout (nn.Dropout): Dropout layer
    """

    def __init__(self, config):
        """Initialize the position-wise feed-forward network.

        Args:
            config: Configuration object containing:
                - hidden_size: Size of hidden states
                - intermediate_size: Size of intermediate layer
                - hidden_dropout_prob: Dropout probability
                - layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward transformation.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, seq_length, hidden_size)
        """
        hidden_states_inner = self.dense1(hidden_states)
        hidden_states_inner = self.intermediate_act_fn(hidden_states_inner)
        hidden_states_inner = self.dense2(hidden_states_inner)
        hidden_states_inner = self.dropout(hidden_states_inner)
        hidden_states = self.LayerNorm(hidden_states + hidden_states_inner)
        return hidden_states


class TransformerLayer(nn.Module):
    """Transformer layer with multi-head attention and position-wise FFN.

    This module implements a standard transformer encoder layer with:
        1. Multi-head self-attention with relative position encoding
        2. Position-wise feed-forward network
        3. Residual connections and layer normalization

    Attributes:
        attention (MultiHeadAttention): Multi-head attention module
        ffn (PositionWiseFFN): Position-wise feed-forward network
    """

    def __init__(self, config):
        """Initialize the transformer layer.

        Args:
            config: Configuration object for attention and FFN modules
        """
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = PositionWiseFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input through attention and feed-forward layers.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional attention mask tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output: Transformed tensor of shape (batch_size, seq_length, hidden_size)
                - attention_probs: Attention probabilities from self-attention layer
        """
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        layer_output = self.ffn(attention_output)
        return layer_output, attention_probs


class AdvancedPooling(nn.Module):
    """Advanced pooling mechanism combining multiple pooling strategies.

    This module implements a sophisticated pooling mechanism that combines:
        1. Mean pooling across sequence length
        2. Max pooling across sequence length
        3. Attention-weighted pooling
        4. Dense layer to combine different pooling results
        5. Layer normalization and dropout

    Attributes:
        dense (nn.Linear): Linear layer to combine pooling results
        dropout (nn.Dropout): Dropout layer
        LayerNorm (nn.LayerNorm): Layer normalization
    """

    def __init__(self, config):
        """Initialize the advanced pooling module.

        Args:
            config: Configuration object containing:
                - hidden_size: Size of hidden states
                - hidden_dropout_prob: Dropout probability
                - layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply advanced pooling to sequence of hidden states.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Pooled representation of shape (batch_size, hidden_size)
        """
        # Mean pooling
        mean_pool = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
        mean_pool = mean_pool / torch.sum(attention_mask, dim=1, keepdim=True)

        # Max pooling
        max_pool = torch.max(
            hidden_states + (1 - attention_mask.unsqueeze(-1)) * -1e9, dim=1
        )[0]

        # First token ([CLS]) pooling
        first_token = hidden_states[:, 0]

        # Combine all pooling strategies
        pooled = torch.cat([mean_pool, max_pool, first_token], dim=-1)
        pooled = self.dense(pooled)
        pooled = self.dropout(pooled)
        pooled = self.LayerNorm(pooled)

        return pooled


class AdvancedDocClassificationModel(nn.Module):
    """Advanced document classification model with sophisticated transformer architecture.

    This model uses a pre-trained LLM (DeBERTa-v3) as the backbone and adds multiple
    transformer layers with advanced attention mechanisms and pooling strategies.
    Features include:
        1. Pre-trained language model backbone
        2. Additional transformer layers with relative position encoding
        3. Advanced pooling mechanism
        4. Multiple classification heads for intermediate supervision
        5. Comprehensive configuration management

    Attributes:
        pretrained_model_name (str): Name of the pre-trained model
        num_hidden_layers (int): Number of additional transformer layers
        config (AutoConfig): Model configuration
        base_model (AutoModel): Pre-trained language model
        transformer_layers (nn.ModuleList): List of additional transformer layers
        pooling (AdvancedPooling): Advanced pooling module
        classifiers (nn.ModuleList): List of classification heads
    """

    def __init__(
        self,
        n_doc_labels: int,
        p_dropout: float = 0.1,
        num_hidden_layers: int = 2,
        pretrained_model_name: str = "microsoft/deberta-v3-base",
    ):
        """Initialize the advanced document classification model.

        Args:
            n_doc_labels: Number of target document classes
            p_dropout: Dropout probability (default: 0.1)
            num_hidden_layers: Number of additional transformer layers (default: 2)
            pretrained_model_name: Name of the pre-trained model
                (default: "microsoft/deberta-v3-base")

        Raises:
            ValueError: If n_doc_labels <= 0 or p_dropout not in [0,1]
        """
        if n_doc_labels <= 0:
            raise ValueError("n_doc_labels must be positive")
        if not 0 <= p_dropout <= 1:
            raise ValueError("p_dropout must be between 0 and 1")

        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.num_hidden_layers = num_hidden_layers

        # Load pre-trained model and config
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.config.hidden_dropout_prob = p_dropout
        self.config.attention_probs_dropout_prob = p_dropout

        # Initialize the base model
        self.base_model = AutoModel.from_pretrained(
            pretrained_model_name, config=self.config
        )

        # Add additional transformer layers
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(self.config) for _ in range(num_hidden_layers)]
        )

        # Advanced pooling
        self.pooling = AdvancedPooling(self.config)

        # Classification heads (including intermediate supervision)
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, n_doc_labels)
                for _ in range(num_hidden_layers + 1)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - logits: Final classification logits
                - intermediate_logits: List of intermediate logits for each layer
                - attention_probs: List of attention probabilities from each layer
        """
        # Validate input dimensions
        if input_ids.shape != attention_mask.shape:
            raise RuntimeError(
                f"Mismatched dimensions: input_ids shape {input_ids.shape} != "
                f"attention_mask shape {attention_mask.shape}"
            )

        # Create extended attention mask for transformer layers
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # Get base model outputs
        base_outputs = self.base_model(input_ids, attention_mask)
        hidden_states = base_outputs.last_hidden_state

        # Initialize lists for intermediate outputs
        all_logits = []
        all_attention_probs = []

        # Get logits from base model outputs
        pooled_output = self.pooling(hidden_states, attention_mask)
        layer_logits = self.classifiers[0](pooled_output)
        all_logits.append(layer_logits)

        # Pass through additional transformer layers
        for i, layer in enumerate(self.transformer_layers):
            hidden_states, attention_probs = layer(
                hidden_states, extended_attention_mask
            )
            pooled_output = self.pooling(hidden_states, attention_mask)
            layer_logits = self.classifiers[i + 1](pooled_output)

            all_logits.append(layer_logits)
            all_attention_probs.append(attention_probs)

        # Return final logits and intermediate outputs
        return {
            "logits": all_logits[-1],  # Final layer logits
            "intermediate_logits": all_logits[:-1],  # Intermediate layer logits
            "attention_probs": all_attention_probs,  # Attention probabilities
        }

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving and reproduction.

        Returns:
            Dict[str, Any]: Configuration dictionary containing:
                - n_doc_labels: Number of document classes
                - p_dropout: Dropout probability
                - num_hidden_layers: Number of transformer layers
                - pretrained_model_name: Name of pre-trained model
        """
        return {
            "n_doc_labels": self.classifiers[0].out_features,
            "p_dropout": self.config.hidden_dropout_prob,
            "num_hidden_layers": self.num_hidden_layers,
            "pretrained_model_name": self.pretrained_model_name,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_attention_heads": self.config.num_attention_heads,
        }
