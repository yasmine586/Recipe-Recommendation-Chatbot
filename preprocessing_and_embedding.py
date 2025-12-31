import pandas as pd
import ast
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document 
import pickle
def simple_preprocess(df, output_csv):
    
    
    def safe_parse(cell):
        try:
            return ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return []

    df['ingredients_parsed'] = df['NER'].apply(safe_parse)

    
    df['NER_flat'] = df['ingredients_parsed'].apply(lambda x: ' '.join(x).lower())

   
    df = df[df['NER_flat'].str.len() > 20]

    
    df[['title','directions','NER_flat','link','ingredients']].to_csv(output_csv, index=False)
def build_recipe_vector_store(df, save_path: str = "faiss_recipe_index"):
    df["combined_text"] = (
        'Recipe: "' + df["title"].fillna("") + '" '
        'Ingredients: ' + df["NER_flat"].fillna(""))
 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",encode_kwargs={"normalize_embeddings": True})
    
    documents = []
    
    print("Preparing documents...")
    for _, row in df.iterrows():
        
        page_content = row["combined_text"]
        
        metadata = {
            "title": row["title"],
            "ingredients": row["ingredients"],
            "directions": row["directions"],
            "link": row["link"],
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    
    print(f"Embedding {len(documents)} recipes... (this may take a minute)")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    
    vector_store.save_local(save_path)
    
    
    return vector_store

if __name__ == '__main__':
    df = pd.read_csv("recipes_preprocessed_ner.csv",nrows=100000)

    #simple_preprocess('recipes_data.csv', 'recipes_preprocessed_ner.csv')
    build_recipe_vector_store(df)

