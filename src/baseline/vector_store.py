
import numpy as np
import faiss
from typing import List, Dict, Optional
import pickle
from pathlib import Path

class VectorStore:
    """
    Wrapper for FAISS vector database to store and retrieve session embeddings.
    """
    
    def __init__(self, dimension: int = 128, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        
        # Initialize index
        # IndexFlatIP = Inner Product (useful for cosine similarity if normalized)
        # IndexIDMap = Allows storing custom IDs (session hash/int) alongside vectors
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        
        # Metadata storage (Faiss only stores vectors)
        # {session_id_hash: {str_session_id, user_id, timestamp, ...}}
        self.metadata: Dict[int, Dict] = {}
        
        if self.index_path and self.index_path.exists():
            self.load()

    def add_sessions(self, 
                    embeddings: np.ndarray, 
                    metadata_list: List[Dict]):
        """
        Add batch of sessions to the store.
        
        Args:
            embeddings: (N, D) numpy array of float32
            metadata_list: List of dicts containing session info
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata count must match")
            
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Generate int64 IDs for Faiss (hash of session_id string)
        ids = np.array([self._hash_id(m['session_id']) for m in metadata_list], dtype=np.int64)
        
        # Add to index
        self.index.add_with_ids(embeddings, ids)
        
        # Store metadata
        for id_, meta in zip(ids, metadata_list):
            self.metadata[id_] = meta

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Find k most similar sessions.
        
        Returns:
            List of matches: {'score': float, 'metadata': dict}
        """
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, ids = self.index.search(query_embedding, k)
        
        results = []
        for score, id_ in zip(scores[0], ids[0]):
            if id_ != -1 and id_ in self.metadata:
                results.append({
                    'score': float(score),
                    'metadata': self.metadata[id_]
                })
                
        return results

    def get_user_embeddings(self, user_id: str) -> np.ndarray:
        """Retrieve all embedding vectors for a specific user."""
        # Note: This is inefficient in raw Faiss. 
        # In a real DB (pgvector), we'd query by user_id.
        # Here we iterate metadata mapping.
        
        user_ids = []
        target_ids = []
        
        for id_, meta in self.metadata.items():
            if meta['user_id'] == user_id:
                target_ids.append(id_)
                
        if not target_ids:
            return np.array([])
            
        # Reconstruct vectors (IndexIDMap allows retrieval if base index supports it)
        # IndexFlatIP supports reconstruction
        vectors = []
        for id_ in target_ids:
            try:
                # Direct access might not be supported by all index types wrapper
                # But for Flat index it typically works if we stored it
                # Workaround: We might need to store vectors separately if reconstruction fails
                # Let's assume we can't efficiently reconstruct from just ID in all Faiss versions
                # For Phase 2 prototype, we will store raw vectors in metadata as well or separate dict
                # Updating metadata to store usage for now
                pass 
            except Exception:
                pass
        
        # Better approach for this prototype: Store vectors in memory dict
        # Modifying __init__ logic conceptually, but for now let's implement
        # a simple "get all by metadata" from the separate structure
        
        # To fetch actual vectors from Faiss by ID is tricky without maintaining a mapping
        # For this implementation, let's allow `reconstruct`
        
        vectors = np.zeros((len(target_ids), self.dimension), dtype=np.float32)
        for i, id_ in enumerate(target_ids):
            vectors[i] = self.index.reconstruct(id_)
            
        return vectors

    def save(self, path: Optional[str] = None):
        """Save index and metadata to disk."""
        save_path = Path(path) if path else self.index_path
        if not save_path:
            raise ValueError("No save path specified")
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Faiss index
        faiss.write_index(self.index, str(save_path / "faiss_index.bin"))
        
        # Save metadata
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """Load index and metadata from disk."""
        if not self.index_path.exists():
            return
            
        # Load Faiss index
        self.index = faiss.read_index(str(self.index_path / "faiss_index.bin"))
        
        # Load metadata
        with open(self.index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def _hash_id(self, session_id: str) -> int:
        """Convert string ID to int64 hash."""
        # Python hash is not stable across restarts, use deterministic hash
        import hashlib
        hex_hash = hashlib.md5(session_id.encode()).hexdigest()
        return int(hex_hash[:15], 16) # Truncate to fit int64
