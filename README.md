# Recipe-Recommendation-Chatbot
A Hybrid RAG-based Recipe Chatbot using Llama 3.3 and FAISS. Combines semantic search with deterministic constraint filtering for safe, allergy-aware recipe recommendations.
<img width="745" height="391" alt="image" src="https://github.com/user-attachments/assets/2f72e865-1f4b-4a73-90fb-7e399557cd71" />

##  Prerequisites

- Python ≥ 3.10  
- Git  
- [Git LFS](https://git-lfs.github.com/) (for large FAISS index files)  
- Streamlit and other dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```
**FAISS Index Files**

The application relies on pre-built FAISS index files stored in faiss_recipe_index/:

index.faiss

index.pkl

⚠️ These files are large (>100 MB) and are tracked using Git LFS.

If you cloned the repository for the first time, make sure to fetch the LFS files:
```bash
git lfs install
git lfs pull
```
Alternatively, if you don't want to use Git LFS, you can download the FAISS index files from an external source and place them in faiss_recipe_index/.


**Environment Variables**

Create a .env file at the root of the project with your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```
 **Running the Application**

Launch the Streamlit app:
```bash
streamlit run app.py
```
**License & Credits**

Built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), [HuggingFace](https://huggingface.co/) embeddings, and [Groq LLM](https://groq.com/).


FAISS vector index created from recipe data.
