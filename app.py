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
st.set_page_config(page_title="CookBot", layout="wide")

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
        temperature=0.0,
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    
    return vector_store, llm

vector_store, llm = load_heavy_resources()

def extract_constraints_with_llm(query: str, llm) -> dict:
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a constraint extraction system for a recipe assistant.

First, determine if the query is asking for a recipe, recipe recommendations, food suggestions, dishes, meals, or anything related to cooking, eating, or preparing food.
- If it's about recipes, food, cooking, desserts, meals, or includes preferences like dietary restrictions (e.g., "vegan recipes", "dessert without chocolate", "quick vegan pasta", "healthy dinner ideas", "recipes using tomatoes"), set "is_recipe_query": true
- If it's completely unrelated to food or cooking (e.g., "talk about football", "weather today", "who won the game"), set "is_recipe_query": false

Only if "is_recipe_query" is true, extract constraints.
Return ONLY a valid JSON object with this exact structure:

{{
  "is_recipe_query": true,
  "forbidden_ingredients": ["egg", "cheese"],
  "required_ingredients": ["tomato", "pasta"],
  "preferences": ["vegan", "quick", "healthy"]
}}

RULES:
- Use singular base forms (egg, not eggs; tomato, not tomatoes)
- Only extract explicitly mentioned items – do NOT infer or add common substitutions
- For dietary terms like "vegan", add to preferences only (do not auto-add forbidden ingredients)
- For exclusions like "without chocolate", add "chocolate" to forbidden_ingredients
- If nothing matches a field, use empty array []
- If the query refers to previous recipes (e.g., contains "these", "them", "the ones", "add to", "in those"), or is a modification/question about existing results, set "is_recipe_query": false even if it mentions food.
- Return ONLY clean JSON, no markdown, no explanation, no extra text"""),
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
        }
    except Exception as e:
        st.warning(f"Couldn't parse your request clearly. Treating as non-recipe query.")
        return {
            "is_recipe_query": False,
            "forbidden": set(),
            "required": set(),
            "preferences": [],
        }

INTERNAL_RETRIEVAL_K = 200 

def retrieve_candidates(query: str):
    return vector_store.similarity_search(query, k=INTERNAL_RETRIEVAL_K)

def filter_by_constraints(docs: list, constraints: dict) -> list:
    """
    Deterministic filtering post-retrieval for hard constraints like forbidden ingredients.
    Allows partial matches for required ingredients to enable suggestions and substitutions.
    """
    forbidden = constraints['forbidden'].copy()
    preferences = constraints['preferences']
    
    
    if 'vegan' in preferences:
        forbidden.update({'egg', 'milk', 'dairy', 'cheese', 'butter', 'honey', 'meat', 'chicken', 'fish', 'seafood'})
    if 'vegetarian' in preferences:
        forbidden.update({'meat', 'chicken', 'beef', 'pork', 'fish', 'seafood'})
    if 'gluten-free' in preferences or 'gluten free' in preferences:
        forbidden.update({'wheat', 'flour', 'bread', 'pasta', 'barley'})

    filtered = []
    

    for doc in docs:
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

        
        filtered.append(doc)

    return filtered

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Tell me what you want to eat, and I'll find the perfect recipe!"}
    ]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000
    )

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

st.title("🍳 CookBot - Recipe Assistant")
st.caption("Just tell me what you're craving – I'll find the best matching recipes!")

with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.last_context = ""
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_response(query):
    constraints = extract_constraints_with_llm(query, llm)
    context_text = ""
    
    if constraints["is_recipe_query"]:
        with st.spinner("🔍 Looking for the best recipes..."):
            augmented_query = f"{query} {' '.join(constraints['preferences'])}"
            candidate_docs = retrieve_candidates(augmented_query)
            filtered_docs = filter_by_constraints(candidate_docs, constraints)
            final_results = filtered_docs[:NUM_RECIPES_TO_SHOW]
            
            
            if final_results:
                context_text = "HERE ARE THE RECIPES FOUND IN DATABASE:\n"
                for doc in final_results:
                    context_text += f"- Title: {doc.metadata.get('title')}\n"
                    context_text += f"- Ingredients: {doc.metadata.get('ingredients')}\n"
                    context_text += f"- Directions: {doc.metadata.get('directions')}\n\n"
                    context_text += f"- Link: {doc.metadata.get('link', 'N/A')}\n\n"
                
                st.session_state.last_context = context_text
            else:
                st.session_state.last_context = ""  
                return "I couldn't find any recipes that match your request. Please try different ingredients or preferences."
    
    else:
        if st.session_state.last_context:
            context_text = f"THESE ARE THE RECIPES THE USER IS ASKING ABOUT (Answer based ONLY on these):\n{st.session_state.last_context}"
        else:
            context_text = "No recipes loaded yet."

    chat_history = st.session_state.memory.load_memory_variables({})['chat_history']
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Chef AI. 
        
        CONTEXT FROM DATABASE (Recipes - sorted by relevance score, use in this exact order):
        {context}
        
        INSTRUCTIONS:
         - Answer using the recipes in the CONTEXT. You may suggest common substitutions for missing ingredients or note if a recipe only partially matches the requested ingredients, but do NOT create entirely new recipes or add details not based on the CONTEXT or general cooking knowledge.
         - List ALL provided recipes exactly as given, in the order they appear (highest relevance first).
         - For each recipe: Use Markdown with Title, Ingredients (as list), Directions (as steps), and MUST include the source link as [Full Recipe](URL).
         - If follow-up (e.g., 'add chicken?'), explain feasibility or suggest substitutions based on the CONTEXT recipes without creating new ones.
         - If no recipes in CONTEXT, say: "No matching recipes found - try different terms!"
         - Be warm, professional, and concise - no extra fluff.
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
    
    return response.content

if prompt := st.chat_input("Ex: Pasta recipe, then ask 'Can I add chicken?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response_text = generate_response(prompt)
        st.markdown(response_text)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text
    })
    st.session_state.memory.save_context({"input": prompt}, {"output": response_text})