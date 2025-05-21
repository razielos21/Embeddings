import re
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text

# Load Document

def load_docx(file_path):
    doc = docx.Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def load_pdf(file_path):
    text = extract_text(file_path)
    if not text:
        return []
    # Normalize line endings
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace single newlines (line wraps) with spaces, keep double newlines (paragraphs)
    normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', normalized)
    # Split on double newlines to get paragraphs
    paragraphs = [para.strip() for para in normalized.split('\n\n') if para.strip()]
    return paragraphs

def load_document(file_path):
    if file_path.endswith(".docx"):
        return load_docx(file_path)
    elif file_path.endswith(".pdf"):
        return load_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Only .docx and .pdf are supported.")

# Sentence Splitter

def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

# Split with Overlap

def split_with_overlap(paragraphs, chunk_size=5, overlap=2):
    segments = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        n = len(sentences)
        if n == 0:
            continue
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunk = sentences[start:end]
            if chunk:
                segments.append(" ".join(chunk))
            if end == n:
                break
            start += chunk_size - overlap
    return segments

# Embedder

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, segments):
        return self.model.encode(segments, convert_to_numpy=True)

# FAISS DB

class FaissStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings, segments):
        self.index.add(embeddings)
        self.texts.extend(segments)

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(np.array([query_embedding]), top_k)
        return [(D[0][i], self.texts[I[0][i]]) for i in range(top_k)]

# Main 

def main():
    file_path = "examples/test.docx"
    print(f"Loading: {file_path}")
    paragraphs = load_document(file_path)
    print(f"Found {len(paragraphs)} paragraphs")

    segments = split_with_overlap(paragraphs)
    print(f"Created {len(segments)} segments")

    embedder = Embedder()
    embeddings = embedder.encode(segments)
    print(f"Generated {len(embeddings)} embeddings")

    # Store in database
    db = FaissStore(dimension=embeddings.shape[1])
    db.add(embeddings, segments)
    print("\nEmbeddings stored in FAISS.")

    print("\n--- Segment Embeddings---")
    for i, embedding in enumerate(embeddings):
        result = db.search(embedding, top_k=1)[0]
        distance, retrieved_segment = result

        print(f"\nSegment {i+1}: {retrieved_segment[:100]}{'...' if len(retrieved_segment) > 100 else ''}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding vector: \n{embedding}")

if __name__ == "__main__":
    main()