"""Semantic router for Yaz fact-atoms (the keystone of Yaz + Engram).

Maps a prompt string -> a fact-atom id using FROZEN Engram (MiniLM, 384-d) embeddings,
so paraphrases of the same fact route to the same atom. Two routers, both over the same
frozen embeddings:

  - FROZEN centroid: atom key = mean embedding of a country's TRAIN-template prompt
    prefixes; route = nearest key by cosine. Zero learned params.
  - LEARNED linear: a trainable Linear(384 -> n_country) trained (CE) on the same
    TRAIN-template embeddings; route = argmax. Tests whether a learned projection over
    frozen embeddings generalizes to held-out phrasings.

Country i (in `country_order`) owns fact-atom id i (matches train_gen's c2i ordering),
so a router that returns country index returns the atom id directly.

Leak-free: keys/head use ONLY train templates; held-out templates never enter fitting.
MiniLM is frozen, never fine-tuned on these facts.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("YAZ_EMBEDDER_PATH", ""))
from engram import Embedder  # noqa: E402


class SemanticRouter:
    def __init__(self, country_order, train_templates, prefer="auto"):
        self.country_order = list(country_order)          # atom id == index here
        self.train_templates = list(train_templates)
        self.emb = Embedder(prefer=prefer)
        assert self.emb.mode == "st", f"need semantic embeddings, got mode={self.emb.mode}"
        self.dim = int(self.emb.dim)
        self._cache: dict[str, np.ndarray] = {}
        self.centroids: np.ndarray | None = None          # (n, dim) unit-norm
        self.head: nn.Linear | None = None

    # ---- embedding (cached, deterministic) ----
    def embed(self, prompt: str) -> np.ndarray:
        v = self._cache.get(prompt)
        if v is None:
            v = self.emb.encode_one(prompt).astype(np.float32)  # L2-normed
            self._cache[prompt] = v
        return v

    def _train_matrix(self):
        """Returns (X, y): X=(n*T, dim) train-prefix embeddings, y=(n*T,) country idx."""
        X, y = [], []
        for ci, c in enumerate(self.country_order):
            for t in self.train_templates:
                X.append(self.embed(t.format(C=c)))
                y.append(ci)
        return np.stack(X), np.array(y, dtype=np.int64)

    # ---- frozen centroid router ----
    def build_centroids(self):
        n = len(self.country_order)
        cent = np.zeros((n, self.dim), dtype=np.float32)
        for ci, c in enumerate(self.country_order):
            vs = np.stack([self.embed(t.format(C=c)) for t in self.train_templates])
            m = vs.mean(0)
            cent[ci] = m / (np.linalg.norm(m) + 1e-8)
        self.centroids = cent
        return self

    def route_frozen(self, prompt: str) -> int:
        v = self.embed(prompt)                 # unit-norm
        return int((self.centroids @ v).argmax())   # nearest centroid by cosine

    # ---- learned linear router ----
    def train_head(self, steps=400, lr=1e-2, seed=2026):
        torch.manual_seed(seed)
        X, y = self._train_matrix()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        head = nn.Linear(self.dim, len(self.country_order))
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(head(Xt), yt)
            loss.backward()
            opt.step()
        head.eval()
        self.head = head
        with torch.no_grad():
            tr_acc = float((head(Xt).argmax(1) == yt).float().mean())
        return tr_acc

    def route_learned(self, prompt: str) -> int:
        v = torch.from_numpy(self.embed(prompt))[None]
        with torch.no_grad():
            return int(self.head(v).argmax(1).item())
