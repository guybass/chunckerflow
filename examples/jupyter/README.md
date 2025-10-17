# ChunkFlow Jupyter Notebooks

Interactive tutorials for learning and exploring ChunkFlow's capabilities.

## 📚 Notebooks Overview

### 01 - Getting Started (`01_getting_started.ipynb`)
**Recommended for beginners**

Learn the basics of ChunkFlow:
- Chunking text with different strategies (Fixed, Recursive, Markdown)
- Generating embeddings with HuggingFace
- Evaluating chunk quality with 4 semantic metrics
- Understanding similarity between chunks
- Simulating query-based retrieval

**Time**: ~15 minutes
**Prerequisites**: None

---

### 02 - Strategy Comparison (`02_strategy_comparison.ipynb`)
**Recommended for strategy selection**

Compare multiple chunking strategies:
- Set up and configure 5 different strategies
- Evaluate with multiple metrics
- Rank strategies by performance
- Use ResultsDataFrame for analysis
- Make data-driven decisions

**Time**: ~20 minutes
**Prerequisites**: Notebook 01

---

### 03 - Advanced Metrics (`03_advanced_metrics.ipynb`)
**Recommended for in-depth evaluation**

Deep dive into all 12 evaluation metrics:
- **Semantic Metrics** (4): Coherence, Boundary Quality, Stickiness, Diversity
- **Retrieval Metrics** (4): NDCG@k, Recall@k, Precision@k, MRR
- **RAG Quality Metrics** (4): Context Relevance, Faithfulness, Precision, Recall
- Creating ground truth data
- Interpreting metric scores

**Time**: ~30 minutes
**Prerequisites**: Notebook 01

---

### 04 - Visualization & Analysis (`04_visualization_analysis.ipynb`)
**Recommended for reporting and insights**

Create publication-quality visualizations:
- **7 visualization types**: Heatmaps, bar charts, radar charts, box plots, correlation matrices, scatter plots, dashboards
- ResultsDataFrame analysis methods
- Export to CSV, JSON, Parquet, Excel
- Custom pandas analysis
- Automated reporting

**Time**: ~25 minutes
**Prerequisites**: Notebook 02

---

### 05 - API Usage (`05_api_usage.ipynb`)
**Recommended for application integration**

Learn to use ChunkFlow's REST API:
- API endpoint reference
- Chunking, embedding, evaluation via API
- Strategy comparison via API
- Error handling best practices
- Building a reusable API client
- Docker deployment

**Time**: ~20 minutes
**Prerequisites**: API server running (see below)

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install chunk-flow[huggingface]

# For visualization notebooks (04)
pip install chunk-flow[huggingface,viz]

# For API notebooks (05)
pip install chunk-flow[api,huggingface]

# Everything
pip install chunk-flow[all]
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab

# Navigate to examples/jupyter/ and open any notebook
```

### For API Notebook (05)

Start the ChunkFlow API server before running notebook 05:

```bash
# Terminal 1: Start the API server
uvicorn chunk_flow.api.app:app --reload --port 8000

# Terminal 2: Run Jupyter
jupyter notebook
```

Or use Docker:

```bash
# Start API with Docker
docker-compose up

# Then run Jupyter
jupyter notebook
```

---

## 📖 Learning Paths

### Path 1: Beginner to Expert
1. **01 - Getting Started** (basics)
2. **02 - Strategy Comparison** (choosing strategies)
3. **03 - Advanced Metrics** (evaluation deep dive)
4. **04 - Visualization** (reporting)
5. **05 - API Usage** (integration)

### Path 2: Quick Integration
1. **01 - Getting Started** (basics)
2. **05 - API Usage** (integration)

### Path 3: Research & Analysis
1. **01 - Getting Started** (basics)
2. **02 - Strategy Comparison** (experiments)
3. **03 - Advanced Metrics** (detailed evaluation)
4. **04 - Visualization** (results presentation)

### Path 4: Production Deployment
1. **01 - Getting Started** (basics)
2. **02 - Strategy Comparison** (select best strategy)
3. **05 - API Usage** (deployment)

---

## 💡 Tips

### Running Cells
- **Shift + Enter**: Run current cell and move to next
- **Ctrl/Cmd + Enter**: Run current cell and stay
- **Alt + Enter**: Run current cell and insert new cell below

### Async Code in Jupyter
All notebooks use `await` for async code. Jupyter natively supports top-level `await` in cells.

### GPU Support
For faster embeddings with HuggingFace:

```python
# Change this in any notebook
embedder = EmbeddingProviderFactory.create(
    "huggingface",
    {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cuda",  # Use "cpu" if no GPU
        "normalize": True,
    }
)
```

### Custom Documents
Replace the sample documents in any notebook with your own:

```python
# Your document
document = """
Your text here...
"""
```

---

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Install missing dependencies
```bash
pip install chunk-flow[all]
```

### Issue: API connection error in notebook 05
**Solution**: Start the API server
```bash
uvicorn chunk_flow.api.app:app --reload
```

### Issue: Slow embedding generation
**Solution**: Use GPU or smaller model
```python
# Smaller, faster model
"model": "sentence-transformers/all-MiniLM-L6-v2"

# Or enable GPU
"device": "cuda"
```

### Issue: Out of memory
**Solution**: Reduce chunk size or batch size
```python
# Smaller chunks
"chunk_size": 300  # instead of 1000

# Process fewer chunks at once
chunks[:10]  # instead of all chunks
```

---

## 📊 Example Outputs

Each notebook produces:
- **Notebook 01**: Evaluation metrics, similarity matrices, top retrieval results
- **Notebook 02**: Strategy comparison tables, ranking reports
- **Notebook 03**: Complete metric scores for all 12 metrics
- **Notebook 04**: Publication-quality visualizations (PNG files)
- **Notebook 05**: API response examples, client usage patterns

---

## 🎯 Use Cases

### Research
- Compare chunking strategies for academic papers
- Evaluate metrics for research datasets
- Generate visualizations for publications

### Development
- Prototype chunking pipelines
- Test different configurations
- Debug retrieval issues

### Production
- Select optimal strategy for your data
- Benchmark performance
- Integrate via API

---

## 📚 Additional Resources

- **ChunkFlow Documentation**: See `docs/` in repository
- **API Reference**: http://localhost:8000/docs (when server running)
- **Python Examples**: See `examples/*.py` for non-notebook examples
- **GitHub**: https://github.com/chunkflow/chunk-flow
- **Issues**: https://github.com/chunkflow/chunk-flow/issues

---

## 🤝 Contributing

Found an issue or want to improve a notebook?

1. Report issues: [GitHub Issues](https://github.com/chunkflow/chunk-flow/issues)
2. Submit improvements: See [CONTRIBUTING.md](../../CONTRIBUTING.md)
3. Share your notebooks: We welcome community contributions!

---

## 📝 License

These notebooks are part of ChunkFlow and are released under the MIT License.

---

**Happy Chunking! 🚀**

Built with passion for the neglected field of text chunking.
