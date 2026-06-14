"""Yaz — CRUD-native byte-level transformer POC.

Core differentiator: the final layer before unembed is a top-k=1 "fact atom"
dictionary. Each fact-atom is a learnable direction in residual space; only
one fires per token. Editing or deleting a single column of W_dec gives a
clean, isolated CRUD operation on the model's predictions.
"""
from yaz.model import YazConfig, YazLM

__all__ = ["YazConfig", "YazLM"]
__version__ = "0.0.1"
