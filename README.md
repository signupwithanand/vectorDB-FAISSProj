# 🔍 FAISS Vector Search Educational App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/signupwithanand/vectordb-faissproj/main/app.py)

An interactive Streamlit application that teaches how **FAISS** (Facebook AI Similarity Search) works for vector-based document retrieval through hands-on experimentation.

![FAISS Demo](https://img.shields.io/badge/Demo-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Live Demo

**[🚀 Try the Live App Here!](https://share.streamlit.io/signupwithanand/vectordb-faissproj/main/app.py)**

## 📸 Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400/667eea/ffffff?text=FAISS+Educational+App)

### Vector Visualization
![Vector Visualization](https://via.placeholder.com/800x400/764ba2/ffffff?text=Interactive+2D+Visualization)

## ✨ Features

### 🎯 Educational Focus
- **Visual Learning**: See how vector embeddings work in real-time
- **Interactive Exploration**: Add documents and search through them
- **Under-the-Hood Insights**: Understand what FAISS is doing behind the scenes

### 🔧 Technical Features
- **Sentence Transformers**: Uses HuggingFace's `all-MiniLM-L6-v2` model for embeddings
- **FAISS Integration**: Efficient vector similarity search with cosine similarity
- **Dynamic Index**: Add documents on-the-fly and see the index update
- **2D Visualization**: PCA/t-SNE reduction to visualize high-dimensional embeddings

### 📱 User Interface
1. **Document Management**: Add/remove documents with live embedding preview
2. **Smart Search**: Query with natural language and see similarity scores
3. **Interactive Plots**: Visualize document clusters and search results
4. **Educational Sidebar**: Learn about FAISS concepts as you explore

## 🚀 Deployment

### Streamlit Cloud Deployment
This app is deployed on Streamlit Cloud and automatically updates from the GitHub repository.

**Live URL**: [https://share.streamlit.io/signupwithanand/vectordb-faissproj/main/app.py](https://share.streamlit.io/signupwithanand/vectordb-faissproj/main/app.py)

### Local Development
```bash
# Clone the repository
git clone https://github.com/signupwithanand/vectorDB-FAISSProj.git
cd vectorDB-FAISSProj

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 💻 Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### 1. Add Documents
- Use the left panel to add text documents
- See the embedding vectors generated in real-time
- Watch the document count grow

### 2. Search
- Enter a search query in the right panel
- Adjust the number of results (k)
- See similarity scores and ranked results

### 3. Visualize
- View the 2D projection of your document embeddings
- When you search, see the query point (green star) and similar documents (red dots)
- Switch between PCA and t-SNE for different visualization perspectives

## Understanding the Results

### Similarity Scores
- **Range**: 0.0 to 1.0 (higher = more similar)
- **Algorithm**: Cosine similarity between normalized vectors
- **Interpretation**: >0.8 = very similar, 0.5-0.8 = somewhat similar, <0.5 = not very similar

### Visualization
- **Blue dots**: Your documents in 2D space
- **Green star**: Your search query
- **Red dots**: Most similar documents found by FAISS
- **Distance**: Closer points are more semantically similar

## Educational Concepts

### What is FAISS?
FAISS is a library for efficient similarity search of dense vectors. It's optimized for:
- **Speed**: Fast nearest neighbor search
- **Scale**: Handle millions of vectors
- **Accuracy**: High-quality similarity results

### The Process
1. **Text → Embeddings**: Convert text to 384-dimensional vectors
2. **Indexing**: Store vectors in FAISS for fast retrieval
3. **Search**: Find vectors most similar to your query
4. **Ranking**: Sort results by similarity score

### Why Vector Search?
- **Semantic Understanding**: Captures meaning, not just keywords
- **Robust**: Works with synonyms, paraphrases, and concepts
- **Flexible**: No need for exact keyword matches

## Code Structure

### Core Functions
- `embed_text()`: Convert text to embeddings using sentence transformers
- `create_faiss_index()`: Build FAISS index from embeddings
- `search_faiss()`: Perform similarity search
- `reduce_dimensions()`: Create 2D visualization using PCA/t-SNE
- `plot_embeddings_interactive()`: Generate interactive plots

### Key Technologies
- **Streamlit**: Web interface framework
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embedding generation
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Dimension reduction algorithms

## Example Workflows

### Basic Document Search
1. Add documents: "The cat sat on the mat", "Dogs are great pets", "Machine learning is fascinating"
2. Search: "pets and animals"
3. Observe: The first two documents score higher than the ML document

### Concept Clustering
1. Add documents from different domains (technology, cooking, sports)
2. Use t-SNE visualization
3. Observe: Documents cluster by topic in the 2D space

### Similarity Analysis
1. Add similar documents with slight variations
2. Search with queries of varying specificity
3. Compare: How similarity scores change with query precision

## Troubleshooting

### Common Issues
- **Slow loading**: Model downloads ~80MB on first run
- **Memory usage**: Large document collections may use significant RAM
- **Visualization**: t-SNE is slower but may show better clusters than PCA

### Performance Tips
- Start with fewer documents to understand the concepts
- Use PCA for faster visualization
- Clear documents periodically to reset the index

## Learning Objectives

After using this app, you should understand:
- How text gets converted to numerical vectors
- Why vector similarity captures semantic meaning
- How FAISS efficiently finds nearest neighbors
- The trade-offs between different similarity metrics
- How dimension reduction affects visualization

## Next Steps

To extend your learning:
- Try different sentence transformer models
- Experiment with FAISS index types (IVF, HNSW)
- Implement different similarity metrics
- Add support for document metadata
- Build a production-ready search system

---

**Happy Learning!** 🚀

This app is designed to make vector search concepts tangible and interactive. Experiment freely and observe how changes affect the results!