import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, ops
from mindspore.common.initializer import Normal
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.utils import GPTConfig


class CausalSelfAttention(nn.Cell):
    """Causal Self-Attention for language modeling"""

    def __init__(self, config: GPTConfig):
        super(CausalSelfAttention, self).__init__()
        assert config.embedding_size % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(config.embedding_size, 3 * config.embedding_size)
        # output projection
        self.c_proj = nn.Dense(config.embedding_size, config.embedding_size)
        # regularization
        self.attn_dropout = nn.Dropout(p=config.dropout_rate)
        self.resid_dropout = nn.Dropout(p=config.dropout_rate)
        self.n_head = config.num_heads
        self.n_emb = config.embedding_size
        self.dropout = config.dropout_rate

        self.bias = Tensor(
            np.tril(np.ones(shape=(config.seq_length, config.seq_length))),
            mstype.float32,
        ).view(1, 1, config.seq_length, config.seq_length)
        self.mask_value = Tensor(np.finfo(np.float32).min)

        self.mul = P.BatchMatMul()
        self.mul_t = P.BatchMatMul(transpose_b=True)

    def construct(self, x: Tensor):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensinality

        # 计算多头注意力的 Q, K, V 矩阵
        q, k, v = self.c_attn(x).split(self.n_emb, axis=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose((0, 2, 1, 3))  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose((0, 2, 1, 3))  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose((0, 2, 1, 3))  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = self.mul_t(q, k) / F.sqrt(F.scalar_to_tensor(v.shape[-1]))
        # 执行注意力 Mask
        b = self.bias[:, :, :T, :T]
        att = att * b + -1e9 * (1 - b)
        att = F.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = self.mul(att, v)
        # re-assemble all head outputs side by side
        y = y.transpose((0, 2, 1, 3))
        y = ops.reshape(y, (B, T, C))

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Cell):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Dense(config.embedding_size, 4 * config.embedding_size)
        self.gelu = nn.GELU()
        self.c_proj = nn.Dense(4 * config.embedding_size,
                               config.embedding_size)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Cell):
    # Transformer Block
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm((config.embedding_size,),)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm((config.embedding_size,),)
        self.mlp = MLP(config)

    def construct(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


class GPT(nn.Cell):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        self.wte = nn.Embedding(
            config.vocab_size, config.embedding_size, embedding_table=Normal(
                0.02)
        )
        self.wpe = nn.Embedding(
            config.seq_length, config.embedding_size, embedding_table=Normal(
                0.02)
        )
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.h = nn.SequentialCell([Block(config)
                                   for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm((config.embedding_size,))

        self.position_ids = F.arange(config.seq_length)

        self.lm_head = nn.Dense(
            config.embedding_size,
            config.vocab_size,
            weight_init=Normal(0.02),
            has_bias=False,
        )

    def construct(self, idx, targets=None):
        b, t = idx.shape  # batch size, sequence length

        pos = self.position_ids[None, :t]  # [t]

        # GPT 的前馈部分代码
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:  # 训练阶段
            x = x.view(-1, x.shape[-1])
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1
            )
        else:  # 推理阶段
            x = x[:, [-1], :].view(-1, x.shape[-1])
            # using list [-1] to preserve the time dim
            logits = self.lm_head(x)
            loss = None

        return logits, loss


class GPTWithLoss(nn.Cell):
    """
    GPT training loss

    Args:
        network: backbone network of GPT2/3

    Inputs:
        input_ids: the tokenized inputs

    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, network):
        super(GPTWithLoss, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, input_ids):
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        logits, loss = self.network(tokens, labels)

        return loss