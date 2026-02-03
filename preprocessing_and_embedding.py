import pandas as pd
import spacy
import ast
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document 
import pickle
nlp = spacy.load("en_core_web_sm")
def extract_ingredients(ingredient_text):
    
   
    
    units = {'c', 'c.', 'cup', 'cups', 'tbsp', 'tsp', 'oz', 'oz.', 'lb', 'lb.', 
             'pkg', 'pkg.', 'jar', 'can', 'package', 'g', 'ml', 'l', 'qt', 'pt',
             'box', 'container', 'carton', 'slices', 'slice','small','large','medium'}
    
    
    ingredient_text = ingredient_text.strip('[]"')
    ingredients = re.split(r'",\s*"', ingredient_text)
    
    result = []
    for ing in ingredients:
        ing = ing.strip('"')
        doc = nlp(ing)
        
        
        nouns = [token.text.lower() for token in doc 
                 if token.pos_ in ['NOUN', 'PROPN','ADJ'] and token.text.lower() not in units]
        
        if nouns:
            
            clean_ingredient = ' '.join(nouns)
            
            clean_ingredient = re.sub(r'[.,;]+', '', clean_ingredient)
            clean_ingredient = clean_ingredient.strip()
            
            if clean_ingredient:
                result.append(clean_ingredient)
    
    
    return ', '.join(result)
def simple_preprocess(df, output_csv):
    
    
    

    df['ingredients_parsed'] = df['ingredients'].apply(extract_ingredients)

    
    

   
    df = df[df['ingredients_parsed'].str.len() > 20]

    
    df[['title','directions','ingredients_parsed','link','ingredients']].to_csv(output_csv, index=False)
def build_recipe_vector_store(df, save_path: str = "faiss_recipe_index"):
    df["combined_text"] = (
        'Recipe: "' + df["title"].fillna("") + '" '
        'Ingredients: ' + df["ingredients_parsed"].fillna(""))
 
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
    df = pd.read_csv("recipes_preprocessed.csv")
    #df=pd.read_csv("recipes_data.csv",nrows=30000)
    #simple_preprocess(df, 'recipes_preprocessed.csv')
    build_recipe_vector_store(df)

