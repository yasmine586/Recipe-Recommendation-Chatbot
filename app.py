import streamlit as st
import pandas as pd
import ast
import os
import re
import json
from langchain_classic.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
NUM_RECIPES_TO_SHOW = 3
st.set_page_config(page_title="Chef AI", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")



@st.cache_resource
def load_heavy_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vector_store_path = "faiss_recipe_index"
    if not os.path.exists(vector_store_path):
        st.error(f"FAISS index not found at '{vector_store_path}'. Run the build script first.")
        st.stop()
    
    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    
    return vector_store, llm


vector_store, llm = load_heavy_resources()


def extract_constraints_with_llm(query: str, llm) -> dict:
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a constraint extraction system for a recipe assistant.

First, determine if the query is asking for a recipe or recipe recommendations.
- If it's clearly about recipes/food/cooking (e.g., "pizza recipe", "vegan pasta", "quick dinner ideas"), set "is_recipe_query": true
- If it's unrelated (e.g., "talk about football", "weather today", "who won the game"), set "is_recipe_query": false

Only if "is_recipe_query" is true, extract constraints.
Return ONLY a valid JSON object with this exact structure:

{{
  "is_recipe_query": true/false,
  "forbidden_ingredients": ["egg", "cheese"],  // base forms, only explicitly mentioned exclusions
  "required_ingredients": ["tomato", "pasta"],  // explicitly mentioned must-have ingredients
  "preferences": ["vegan", "quick", "healthy"], // dietary styles or qualities
  "meal_type": "breakfast/lunch/dinner/dessert/snack/null",
  "cooking_time": "quick/medium/long/null"
}}

RULES:
- Use singular base forms (egg, not eggs; tomato, not tomatoes)
- Only extract explicitly mentioned items – do NOT infer or add common substitutions
- For dietary terms like "vegan", add to preferences only (do not auto-add forbidden ingredients)
- If nothing matches a field, use empty array or null
- Return ONLY clean JSON, no markdown, no explanation"""),
        ("human", "{query}")
    ])
    
    chain = extraction_prompt | llm
    response = chain.invoke({"query": query})
    
    try:
        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE | re.IGNORECASE).strip()
        data = json.loads(content)
        
        return {
            "is_recipe_query": data.get("is_recipe_query", False),
            "forbidden": set(data.get("forbidden_ingredients", [])),
            "required": set(data.get("required_ingredients", [])),
            "preferences": [p.lower() for p in data.get("preferences", [])],
            "meal_type": data.get("meal_type"),
            "cooking_time": data.get("cooking_time")
        }
    except Exception as e:
        st.warning(f"Couldn't parse your request clearly. Treating as recipe query.")
        return {
            "is_recipe_query": False,
            "forbidden": set(),
            "required": set(),
            "preferences": [],
            "meal_type": None,
            "cooking_time": None
        }



INTERNAL_RETRIEVAL_K = 100 


def retrieve_with_scores(query: str):
    docs_and_distances = vector_store.similarity_search_with_score(query, k=INTERNAL_RETRIEVAL_K)
    
    results = []
    for doc, distance in docs_and_distances:
        cosine_sim = 1 - (distance ** 2) / 2
        cosine_sim = max(0.0, min(1.0, cosine_sim))
        cosine_sim = round(cosine_sim, 3)
        
        results.append({
            "doc": doc,
            "faiss_similarity": cosine_sim,
            "faiss_distance": float(distance)
        })
    
    return results


def filter_by_constraints(docs_with_scores: list, constraints: dict) -> list:
    """
    ACADEMIC HIGHLIGHT: Deterministic filtering post-retrieval.
    This ensures safety (allergies) where vector search might fail.
    """
    forbidden = constraints['forbidden'].copy()
    preferences = constraints['preferences']
    
    # Expand Dietary Preferences into specific forbidden ingredients
    if 'vegan' in preferences:
        forbidden.update({'egg', 'milk', 'dairy', 'cheese', 'butter', 'honey', 'meat', 'chicken', 'fish', 'seafood'})
    if 'vegetarian' in preferences:
        forbidden.update({'meat', 'chicken', 'beef', 'pork', 'fish', 'seafood'})
    if 'gluten-free' in preferences or 'gluten free' in preferences:
        forbidden.update({'wheat', 'flour', 'bread', 'pasta', 'barley'})

    filtered = []
    required = constraints['required']

    for item in docs_with_scores:
        doc = item['doc']
        
        
        ingredients = ast.literal_eval(doc.metadata.get('ingredients', '[]'))
        ingredients_text = ' '.join([str(i).lower() for i in ingredients])
        title = doc.metadata.get('title', '').lower()
        searchable_text = f"{ingredients_text} {title}"

        
        has_forbidden = False
        for fb in forbidden:
            
            if re.search(r'\b' + re.escape(fb) + r's?\b', searchable_text, re.IGNORECASE):
                has_forbidden = True
                break
        if has_forbidden: continue

        
        matches = 0
        if required:
            recipe_ingredient_count = len(ingredients)
            matches = 0
    
            for recipe_ing in ingredients:
                recipe_ing_lower = str(recipe_ing).lower()
                for user_ing in required:
                    if re.search(r'\b' + re.escape(user_ing) + r's?\b', 
                       recipe_ing_lower, re.IGNORECASE):
                        matches += 1
                        break
    
                ingredient_match_score = matches / recipe_ingredient_count if recipe_ingredient_count > 0 else 0.0
        else:
            ingredient_match_score = 1.0 # No requirements means perfect match score

        item['ingredient_match'] = ingredient_match_score
        
        
        # We weigh exact ingredient matches higher than semantic vector similarity
        item['final_score'] = (ingredient_match_score * 0.7) + (item['faiss_similarity'] * 0.3)
        
        filtered.append(item)

    
    filtered.sort(key=lambda x: x['final_score'], reverse=True)
    return filtered




if "messages" not in st.session_state:
   
    st.session_state.messages = [
        {"role": "assistant", "content": "👨‍🍳 Hello! I'm Chef AI. Tell me what you want to eat, and I'll find the perfect recipe!"}
    ]

if "memory" not in st.session_state:
   
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000
    )

if "last_context" not in st.session_state:
    
    st.session_state.last_context = ""

st.title("🍳 Chef AI - Recipe Assistant")
st.caption("Just tell me what you're craving – I'll find the best matching recipes!")


with st.sidebar:
    st.header("⚙️ Options")
    
    show_scores = st.checkbox("Show detailed relevance scores", value=False)
    if st.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.last_context = ""
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("score_data") is not None and show_scores:
            st.caption("📊 Analysis :")
            st.dataframe(pd.DataFrame(message["score_data"]), use_container_width=True)


def generate_response(query):
    
    constraints = extract_constraints_with_llm(query, llm)
    score_data=None
    context_text = ""
    
    
    if constraints["is_recipe_query"]:
        with st.spinner("🔍 Looking for the best recipies..."):
            candidate_docs = retrieve_with_scores(query)
            filtered_docs = filter_by_constraints(candidate_docs, constraints)
            final_results = filtered_docs[:NUM_RECIPES_TO_SHOW]
            
            
            if final_results:
                context_text = "HERE ARE THE RECIPES FOUND IN DATABASE:\n"
                for item in final_results:
                    doc = item['doc']
                    context_text += f"- Title: {doc.metadata.get('title')}\n"
                    context_text += f"- Ingredients: {doc.metadata.get('ingredients')}\n"
                    context_text += f"- Directions: {doc.metadata.get('directions')}\n\n"
                    context_text += f"- Link: {doc.metadata.get('link', 'N/A')}\n\n"
                
                
                st.session_state.last_context = context_text
                score_data = []
                for item in final_results:
                    score_data.append({
                        'Recipe': item['doc'].metadata.get('title', 'Unknown'),
                        'Faiss similarity': round(item['faiss_similarity'], 3),
                        
                      
                        'Combined Score with ingredient match': round(item['ingredient_match'] * 0.7 + item['faiss_similarity'] * 0.3, 3)
                    })
            else:
                st.session_state.last_context = ""  
                return "I couldn't find any recipes that match your request. Please try different ingredients or preferences."
    
    
    else:
        if st.session_state.last_context:
            context_text = f"PREVIOUS RECIPES CONTEXT:\n{st.session_state.last_context}"
        else:
            context_text = "No recipes loaded yet."

   
    chat_history = st.session_state.memory.load_memory_variables({})['chat_history']
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Chef AI. 
        
        CONTEXT FROM DATABASE (Recipes):
        {context}
        
        INSTRUCTIONS:
         - Answer only cooking/recipe/food questions.
        - Present recipes beautifully using Markdown.
        - For each recipe, you MUST include the source link provided in the context.
        - Format the link clearly, for example: [Click here to view the full recipe](URL)
        - If the user asks a follow-up question, use the context to answer.
        - Be warm and professional.
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])
    
    chain = final_prompt | llm
    response = chain.invoke({
        "context": context_text, 
        "chat_history": chat_history,
        "question": query
    })
    
    return response.content,score_data


if prompt := st.chat_input("Ex: Pasta recipe, then ask 'Can I add chicken?'"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    
    with st.chat_message("assistant"):
        response_text, new_score_data = generate_response(prompt)
        st.markdown(response_text)
        
        
        if new_score_data and show_scores:
            st.caption("📊 Analysis :")
            st.dataframe(pd.DataFrame(new_score_data), use_container_width=True)
    
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "score_data": new_score_data  
    })
    st.session_state.memory.save_context({"input": prompt}, {"output": response_text})