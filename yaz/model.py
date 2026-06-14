"""Yaz POC architecture.

Standard byte-level causal transformer (3 blocks, d=128) with one twist:
just before unembed, a top-k=1 "fact atom" dictionary projects the
residual into d_dict=512 atoms, picks the single most-activated one,
and adds that atom's learnable decoder vector back into the residual.

This is the CRUD-target layer:
  - W_dec[:, atom_id]  ⟵  edit this single column = edit that fact
  - zero it             ⟵  delete the fact
  - append a new column ⟵  add a fact
The encoder's bias for atom_id controls "is this fact accessible?".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class YazConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    max_seq_len: int = 128
    ffn_expand: int = 4
    dropout: float = 0.0
    # Fact-atom layer:
    d_dict: int = 128        # number of addressable fact atoms
    fact_top_k: int = 1      # v4: strict 1, paired with anti-collapse machinery
    fact_gain: float = 1.0   # multiplier on the fact atom's contribution
    # Phase 9 multi-byte: each atom owns d_phase value vectors; a shared phase
    # head selects which one fires (by within-answer byte offset). d_phase=1 is
    # exactly the single-vector model (byte-identical), so all Phase 1-8 configs
    # are unaffected.
    d_phase: int = 1
    # Semantic re-keying: a learnable per-atom activation gain used by forward_routed
    # so a forced (Engram-routed) atom can DOMINATE the residual — restoring the edit
    # efficacy that a fixed activation=1.0 destroyed (backbone co-memorization). Only
    # created when use_atom_gain=True, so surface (route_atom=None) models are unchanged.
    use_atom_gain: bool = False
    atom_gain_init: float = 1.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: YazConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, d_head)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, H, T, d_head)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class FFN(nn.Module):
    def __init__(self, cfg: YazConfig):
        super().__init__()
        h = cfg.d_model * cfg.ffn_expand
        self.fc1 = nn.Linear(cfg.d_model, h)
        self.fc2 = nn.Linear(h, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: YazConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FactAtomLayer(nn.Module):
    """Top-k=1 atom dictionary. Each atom is an addressable fact slot.

    Shapes:
        encoder  W_enc: (d_dict, d_model)  — pick which atom fires
        decoder  W_dec: (d_model, d_dict)  — its contribution direction
        pre_bias        (d_model,)         — centering offset
    """

    def __init__(self, d_model: int, d_dict: int, top_k: int = 1, d_phase: int = 1,
                 use_atom_gain: bool = False, atom_gain_init: float = 1.0):
        super().__init__()
        if top_k < 1 or top_k > d_dict:
            raise ValueError(f"top_k must be in [1, {d_dict}], got {top_k}")
        if d_phase < 1:
            raise ValueError(f"d_phase must be >= 1, got {d_phase}")
        self.d_model = d_model
        self.d_dict = d_dict
        self.top_k = top_k
        self.d_phase = d_phase
        # Learnable per-atom gain for forward_routed (semantic re-keying). Absent by
        # default so legacy/surface state_dicts are byte-identical.
        self.atom_gain = (nn.Parameter(torch.full((d_dict,), float(atom_gain_init)))
                          if use_atom_gain else None)
        self.W_enc = nn.Linear(d_model, d_dict, bias=True)
        self.W_dec = nn.Linear(d_dict, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        # Tie initial weights — encoder = decoder.T, unit-norm columns.
        with torch.no_grad():
            w = torch.randn(d_dict, d_model)
            w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self.W_dec.weight.copy_(w.t())
            self.W_enc.weight.copy_(w)
            self.W_enc.bias.zero_()
        # Phase 9: extra decoder columns for phases 1..d_phase-1 (phase 0 = W_dec),
        # plus a shared phase head. Created ONLY when d_phase>1 so the d_phase=1
        # state_dict is identical to the legacy model. Extra columns start at zero
        # so an untrained multi-byte model == the single-vector model.
        if d_phase > 1:
            self.W_dec_extra = nn.Parameter(torch.zeros(d_phase - 1, d_model, d_dict))
            self.phase_head = nn.Linear(d_model, d_phase, bias=True)

    def encode(self, x: Tensor) -> Tensor:
        """Returns sparse activations (..., d_dict) with only top_k nonzero entries."""
        z = self.W_enc(x - self.pre_bias)
        z = F.relu(z)
        top_vals, top_idx = z.topk(self.top_k, dim=-1)
        sparse = torch.zeros_like(z).scatter_(-1, top_idx, top_vals)
        return sparse

    def encode_dense(self, x: Tensor) -> Tensor:
        """Returns the PRE-topk gate scores (..., d_dict), for load-balancing loss."""
        return F.relu(self.W_enc(x - self.pre_bias))

    def encode_logits(self, x: Tensor) -> Tensor:
        """Returns the PRE-ReLU encoder logits (..., d_dict), for supervised
        target-atom CE loss (Exp 3). Gradient flows freely through every
        atom dim (no ReLU dead-zone)."""
        return self.W_enc(x - self.pre_bias)

    def decode(self, z: Tensor) -> Tensor:
        return self.W_dec(z) + self.pre_bias

    @torch.no_grad()
    def resurrect(self, dead_idx: Tensor, source: Tensor) -> int:
        """Re-init the encoder row + decoder col for each dead atom from a
        random row of `source` (current residuals, shape (N, d_model)).

        Returns the number of atoms actually resurrected.
        """
        if dead_idx.numel() == 0 or source.numel() == 0:
            return 0
        N = source.size(0)
        # Pick |dead_idx| random source rows
        pick = torch.randint(0, N, (dead_idx.numel(),), device=source.device)
        v = source[pick]                          # (k_dead, d_model)
        norms = v.norm(dim=1, keepdim=True).clamp_min(1e-6)
        v = v / norms                             # unit-norm rows
        self.W_enc.weight[dead_idx] = v
        self.W_dec.weight[:, dead_idx] = v.t()    # decoder col = unit dir
        self.W_enc.bias[dead_idx] = 0.0
        return int(dead_idx.numel())

    def orthogonality_loss(self) -> Tensor:
        """Mean squared off-diagonal of normalized W_dec column Gram matrix.
        0 when all decoder columns are orthonormal; bounded by 1.
        """
        W = self.W_dec.weight                     # (d_model, d_dict)
        norms = W.norm(dim=0, keepdim=True).clamp_min(1e-6)
        Wn = W / norms
        G = Wn.t() @ Wn                           # (d_dict, d_dict)
        eye = torch.eye(self.d_dict, device=G.device, dtype=G.dtype)
        off = G - eye
        return (off ** 2).mean()

    def decode_phased(self, x: Tensor):
        """Phase 9 multi-byte path. Pick the top-1 atom (unchanged), then let the
        shared phase head choose which of the atom's d_phase value vectors fires.

        Returns (out, z_sparse, z_dense, phase_logits). For d_phase=1 this is
        algebraically identical to forward()/decode() with a single column.
        """
        z_dense = self.encode_dense(x)                          # (..., d_dict)
        top_vals, top_idx = z_dense.topk(self.top_k, dim=-1)    # top_k==1
        z_sparse = torch.zeros_like(z_dense).scatter_(-1, top_idx, top_vals)
        a = top_vals[..., 0]                                    # (...) activation
        k = top_idx[..., 0]                                     # (...) atom id
        phase_logits = self.phase_head(x)                       # (..., d_phase)
        v = phase_logits.argmax(dim=-1)                         # (...) chosen phase
        # Stacked decoder: D[0] = legacy W_dec, D[1:] = extra phase columns.
        D = torch.cat([self.W_dec.weight.unsqueeze(0), self.W_dec_extra], dim=0)  # (V, d_model, d_dict)
        shp = k.shape
        col = D[v.reshape(-1), :, k.reshape(-1)]                # (N, d_model)
        contrib = (a.reshape(-1, 1) * col).reshape(*shp, self.d_model)
        out = contrib + self.pre_bias
        return out, z_sparse, z_dense, phase_logits

    def forward(self, x: Tensor, return_dense: bool = False) -> tuple[Tensor, Tensor]:
        z_dense = self.encode_dense(x)
        top_vals, top_idx = z_dense.topk(self.top_k, dim=-1)
        z = torch.zeros_like(z_dense).scatter_(-1, top_idx, top_vals)
        out = self.decode(z)
        if return_dense:
            # Use z_dense so the LB-loss has gradient through W_enc.
            return out, z, z_dense
        return out, z

    def forward_routed(self, x: Tensor, route_atom: Tensor, route_pos: Tensor):
        """SEMANTIC RE-KEYING path. At positions where `route_pos` is True, FORCE the
        atom given by `route_atom` (one per sequence) to fire with activation 1.0 instead
        of the learned argmax(ReLU(W_enc·…)) selection. Elsewhere, learned routing is kept
        (so the language prefix is unaffected and W_enc still trains on non-fact tokens).

        This decouples WHICH atom fires from the byte-transformer surface activations:
        the caller picks the atom via a frozen semantic embedding (Engram), so paraphrases
        route to the same atom. W_dec / pre_bias / the additive contribution / CRUD are
        unchanged — only the selection is overridden.

        route_atom: (B,) int64 atom ids.  route_pos: (B, T) bool.
        Returns (out, z_sparse, z_dense) like forward(return_dense=True).
        """
        z_dense = self.encode_dense(x)                                   # (B,T,d_dict)
        top_vals, top_idx = z_dense.topk(self.top_k, dim=-1)
        z = torch.zeros_like(z_dense).scatter_(-1, top_idx, top_vals)    # learned top-1
        B, T, _ = z.shape
        forced = torch.zeros_like(z)
        idx = route_atom.view(B, 1, 1).expand(B, T, 1)
        if self.atom_gain is not None:
            # learnable per-atom magnitude so the forced atom can dominate the residual
            gain = self.atom_gain[route_atom].view(B, 1, 1).expand(B, T, 1)
            forced.scatter_(-1, idx, gain)
        else:
            forced.scatter_(-1, idx, 1.0)                                # legacy: act=1.0
        z = torch.where(route_pos.unsqueeze(-1), forced, z)
        out = self.decode(z)
        return out, z, z_dense


class YazLM(nn.Module):
    def __init__(self, cfg: YazConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.fact_layer = FactAtomLayer(cfg.d_model, cfg.d_dict, top_k=cfg.fact_top_k,
                                        d_phase=cfg.d_phase, use_atom_gain=cfg.use_atom_gain,
                                        atom_gain_init=cfg.atom_gain_init)
        # Tied embedding for unembed.
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids: Tensor, return_fact_z: bool = False, return_dense: bool = False,
                return_atoms_only: bool = False, return_phase: bool = False,
                route_atom: Tensor | None = None, route_pos: Tensor | None = None):
        """ids: (B, T) of token ids.

        Returns logits (B, T, vocab). Optionally also returns the sparse
        fact-z (for inspection / CRUD address lookup), the dense
        pre-topk gate scores (for the load-balancing loss in training),
        the atoms-only logits (unembed of fact_contrib alone,
        used by the v5 atoms-only auxiliary loss), and/or the per-position
        phase_logits (Phase 9 multi-byte; appended last when return_phase=True).
        """
        B, T = ids.shape
        assert T <= self.cfg.max_seq_len, f"T={T} > max_seq_len={self.cfg.max_seq_len}"
        pos = torch.arange(T, device=ids.device)
        x = self.tok_embed(ids) + self.pos_embed(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)
        phase_logits = None
        if route_atom is not None and self.cfg.d_phase == 1:
            # SEMANTIC RE-KEYING path. Force the caller-supplied atom (chosen by an
            # Engram embedding) at route_pos; default route_pos = last position only.
            if route_pos is None:
                route_pos = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
                route_pos[:, -1] = True
            fact_contrib, fact_z, fact_dense = self.fact_layer.forward_routed(x, route_atom, route_pos)
        elif self.cfg.d_phase > 1:
            # Phase 9 multi-byte path. d_phase=1 NEVER reaches here, so the legacy
            # path below is byte-identical for all Phase 1-8 models.
            fact_contrib, fact_z, fact_dense, phase_logits = self.fact_layer.decode_phased(x)
        elif return_dense:
            fact_contrib, fact_z, fact_dense = self.fact_layer(x, return_dense=True)
        else:
            fact_contrib, fact_z = self.fact_layer(x)
        # Additive fact: the transformer's residual carries language / style;
        # the fact layer contributes one or more learned directions per token.
        # Edits to W_dec[:, k] map onto next-byte logits via unembed; the
        # transformer's contribution is unaffected. CRUD-safe by construction.
        x_full = x + self.cfg.fact_gain * fact_contrib
        logits = self.unembed(x_full)
        # v5 atoms-only path: pretend the transformer contributed nothing,
        # only the fact layer's atom-decoder output (which already includes
        # pre_bias from FactAtomLayer.decode). Training on this with CE on
        # fact tokens forces the model to route fact prediction THROUGH the
        # atom dictionary instead of co-memorizing in the transformer weights.
        if return_atoms_only:
            atoms_only_logits = self.unembed(self.cfg.fact_gain * fact_contrib)
            if return_dense:
                out = (logits, fact_z, fact_dense, atoms_only_logits)
            elif return_fact_z:
                out = (logits, fact_z, atoms_only_logits)
            else:
                out = (logits, atoms_only_logits)
            return out + (phase_logits,) if return_phase else out
        if return_dense:
            out = (logits, fact_z, fact_dense)
            return out + (phase_logits,) if return_phase else out
        if return_fact_z:
            out = (logits, fact_z)
            return out + (phase_logits,) if return_phase else out
        return (logits, phase_logits) if return_phase else logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def greedy_generate(model: YazLM, prompt_ids: Tensor, n_new: int, stop_id: int | None = None) -> Tensor:
    """Greedy generation. prompt_ids: (1, T0)."""
    model.eval()
    out = prompt_ids
    max_ctx = model.cfg.max_seq_len
    for _ in range(n_new):
        ctx = out if out.shape[1] <= max_ctx else out[:, -max_ctx:]
        logits = model(ctx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if stop_id is not None and int(nxt.item()) == stop_id:
            break
    return out
