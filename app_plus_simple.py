import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import ChatPromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Chef AI (With Scores)", layout="wide")


@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.load_local(
        "faiss_recipe_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )
    memory = ConversationBufferMemory(return_messages=True)
    return vector_store, llm, memory

vector_store, llm, memory = load_resources()

# --- 2. SEARCH FUNCTION (Exposes Scores) ---
def search_with_scores(query, k=3):
   
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    
    results = []
    for doc, score in docs_and_scores:
       
        similarity = 1 - (score / 2) 
        results.append({
            "doc": doc,
            "score": round(max(similarity, 0), 2), 
            "raw_dist": round(score, 3)
        })
    return results


with st.sidebar:
    st.header("⚙️ Settings")
    show_scores = st.checkbox("Show Relevance Scores", value=True)
    if st.button("Clear Chat"):
        memory.clear()
        st.session_state.messages = []
        st.rerun()


st.title("👨‍🍳 Chef AI")
st.caption("Ask me about recipes! Toggle the sidebar to see how I rank them.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What are you craving?"}]

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If we saved score data in the message history, display it
        if "scores" in msg and show_scores:
            st.dataframe(msg["scores"], use_container_width=True)


if prompt := st.chat_input("Ex: Spicy chicken pasta"):
    
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner("Searching & Thinking..."):
            
            
            search_results = search_with_scores(prompt)
            
            
            context_text = ""
            score_data_for_ui = []
            
            for res in search_results:
                doc = res['doc']
                
                context_text += f"""
                - Recipe: {doc.metadata.get('title')}
                - Ingredients: {doc.metadata.get('ingredients')}
                - Directions: {doc.metadata.get('directions')}
                - Link: {doc.metadata.get('link')}
                ---------------------
                """
                
                score_data_for_ui.append({
                    "Recipe": doc.metadata.get('title'),
                    "Similarity Score": res['score'],
                    "L2 Distance": res['raw_dist']
                })

            
            history = memory.load_memory_variables({})['history']
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a Chef AI. Use the provided recipes to answer the user.
                
                RECIPES FOUND:
                {context}
                
                INSTRUCTIONS:
                1. Suggest recipes from the list above.
                2. ALWAYS include the ingredients and the Source Link.
                3. If the user's request is not in the recipes, politely say so.
                """),
                ("placeholder", "{chat_history}"),
                ("human", "{question}")
            ])
            
            chain = prompt_template | llm
            response = chain.invoke({
                "context": context_text,
                "chat_history": history,
                "question": prompt
            })
            
            st.markdown(response.content)
            
            # C. SHOW SCORES (Optional)
            if show_scores:
                st.caption("🔍 Search Relevance Metrics")
                st.dataframe(pd.DataFrame(score_data_for_ui), use_container_width=True)

    # 3. Save to History
    memory.save_context({"input": prompt}, {"output": response.content})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response.content,
        "scores": pd.DataFrame(score_data_for_ui) 
    })