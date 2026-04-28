#!/usr/bin/env python3
"""
Shared embedding cache for all experiment scripts.

Saves/loads pre-computed embeddings to disk as .npy files.
Avoids redundant encoding across experiments.

Cache structure:
    data/embeddings_cache/{model_name}/{task_name}.npz
        - arrays: 'embeddings' (float32), optionally 'texts_hash' (int64)

Usage:
    from embedding_cache import EmbeddingCache

    cache = EmbeddingCache('data/embeddings_cache')
    embs = cache.get_or_compute(model_name, task_name, texts, model, device)
"""

import os
import hashlib
import numpy as np
from tqdm import tqdm


class EmbeddingCache:
    def __init__(self, cache_dir='data/embeddings_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, model_name, task_name):
        safe_model = model_name.replace('/', '_')
        safe_task = task_name.replace('/', '_')
        model_dir = os.path.join(self.cache_dir, safe_model)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, f"{safe_task}.npz")

    def _texts_hash(self, texts):
        """Quick hash of text list for cache invalidation."""
        h = hashlib.md5()
        for t in texts[:100]:  # Sample first 100 for speed
            h.update(t.encode('utf-8')[:200])
        h.update(str(len(texts)).encode())
        return int(h.hexdigest()[:16], 16)

    def has(self, model_name, task_name):
        return os.path.exists(self._cache_path(model_name, task_name))

    def load(self, model_name, task_name):
        """Load cached embeddings. Returns numpy array or None."""
        path = self._cache_path(model_name, task_name)
        if not os.path.exists(path):
            return None
        data = np.load(path)
        return data['embeddings']

    def save(self, model_name, task_name, embeddings, texts=None):
        """Save embeddings to cache."""
        path = self._cache_path(model_name, task_name)
        save_dict = {'embeddings': embeddings}
        if texts is not None:
            save_dict['texts_hash'] = np.array([self._texts_hash(texts)])
        np.savez_compressed(path, **save_dict)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    Cached: {path} ({size_mb:.1f} MB)")
        return path

    def get_or_compute(self, model_name, task_name, texts, model, device='cuda:0',
                       batch_size=None, show_progress=True):
        """Load from cache or compute and cache embeddings.

        Args:
            model_name: Model identifier string
            task_name: Task identifier string
            texts: List of strings to encode
            model: SentenceTransformer model instance
            device: Device string
            batch_size: Override batch size (default: adaptive)
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), dim)
        """
        # Try loading from cache
        embs = self.load(model_name, task_name)
        if embs is not None:
            if len(embs) == len(texts):
                print(f"    Loaded cached embeddings: {embs.shape}")
                return embs
            else:
                print(f"    Cache size mismatch: cached={len(embs)}, texts={len(texts)}. Recomputing.")

        # Compute
        print(f"    Encoding {len(texts)} texts...")
        if batch_size is None:
            batch_size = min(64, max(8, 10000 // max(len(texts) // 64, 1)))

        encoded = model.encode(texts, convert_to_tensor=False,
                               show_progress_bar=show_progress, batch_size=batch_size)
        embs = np.array(encoded, dtype=np.float32)

        # Save to cache
        self.save(model_name, task_name, embs, texts)

        return embs

    def get_cache_stats(self):
        """Report cache size and entries."""
        total_size = 0
        entries = []
        for model_dir in os.listdir(self.cache_dir):
            model_path = os.path.join(self.cache_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            for f in os.listdir(model_path):
                if f.endswith('.npz'):
                    fp = os.path.join(model_path, f)
                    size_mb = os.path.getsize(fp) / (1024 * 1024)
                    total_size += size_mb
                    entries.append({
                        'model': model_dir,
                        'task': f[:-4],
                        'size_mb': size_mb,
                    })
        return {'total_size_mb': total_size, 'entries': entries}
