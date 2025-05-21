# Embeddings Segmenter

Python tool to segment text documents (DOCX and PDF) into semantically meaningful chunks, generate embeddings for each segment using a modern transformer model, and store/retrieve these embeddings efficiently using a FAISS vector database.

---

## Features

- **Supports DOCX and PDF files**  
  Extracts and normalizes text from both Microsoft Word and PDF documents.

- **Paragraph and Sentence Splitting**  
  Splits documents into paragraphs and further into sentences for fine-grained segmentation.

- **Overlapping Segments**  
  Segments are created with configurable chunk size and overlap to preserve context.

- **Modern Embedding Model**  
  Uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for fast, high-quality embeddings.

- **Efficient Vector Search**  
  Stores embeddings in a FAISS index for fast similarity search and retrieval.

---

## Requirements

- Python 3.7+
- [sentence-transformers](https://www.sbert.net/)
- [faiss-cpu](https://github.com/facebookresearch/faiss)
- [python-docx](https://python-docx.readthedocs.io/)
- [pdfminer.six](https://pdfminersix.readthedocs.io/)

Install dependencies with:

```bash
pip install sentence-transformers faiss-cpu python-docx pdfminer.six numpy
```

---

## Usage

1. **Place your DOCX or PDF file in the `examples/` folder.**
2. **Edit the `file_path` in `main()` if needed.**
3. **Run the script:**

```bash
python text_segmenter.py
```

---

## How It Works

1. **Document Loading**
    - `load_docx(file_path)`: Loads and cleans paragraphs from a DOCX file.
    - `load_pdf(file_path)`: Extracts and normalizes text from a PDF, preserving paragraph structure.

2. **Segmentation**
    - `split_with_overlap(paragraphs, chunk_size=5, overlap=2)`: Splits paragraphs into overlapping sentence segments.

3. **Embedding**
    - `Embedder`: Uses a transformer model to encode each segment into a vector.

4. **Vector Storage & Search**
    - `FaissStore`: Stores embeddings and segments in a FAISS index for efficient similarity search.

5. **Main Script**
    - Loads the document, segments it, generates embeddings, stores them in FAISS, and demonstrates retrieval.

---

## Example Output

```
Loading: examples/test.docx
Found 6 paragraphs
Created 6 segments
Generated 6 embeddings

Embeddings stored in FAISS.

--- Segment Embeddings---

Segment 1: This is the first segment of text...
Embedding shape: (384,)
Embedding vector:
[ 0.123  0.456 ... ]
```

---

## Customization

- **Change chunk size or overlap:**  
  Edit the parameters in `split_with_overlap()`.

- **Use a different embedding model:**  
  Change the `model_name` in the `Embedder` class.

---

## License

MIT License

---

## Acknowledgments

- [FAISS by Facebook AI Research](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [pdfminer.six](https://pdfminersix.readthedocs.io/)
- [python-docx](https://python-docx.readthedocs.io/)
