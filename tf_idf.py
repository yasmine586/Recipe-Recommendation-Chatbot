import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

df = pd.read_csv("recipes_preprocessed_ner.csv",nrows=10000)
df["combined_text"] = (
    'Recipe: "' + df["title"].fillna("") + '" '
    'Ingredients: ' + df["NER_flat"].fillna("")
)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),      # Unigrams + bigrams
    min_df=2,                # Ignore terms in < 2 recipes
    max_df=0.85,             # Ignore terms in > 85% of recipes
    stop_words=None,         # Already cleaned
    max_features=5000        # Limit vocabulary size
)

# Fit and transform the recipe corpus
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# ========================
# 3. SEARCH WITH USER QUERY
# ========================

def search_recipes_tfidf(user_query, top_n=10):
    """
    Search recipes using TF-IDF based on user's ingredients or query.
    
    Args:
        user_query: str, e.g., "chicken cheese pasta"
        top_n: int, number of results to return
    
    Returns:
        DataFrame with top matching recipes and their scores
    """
    # Transform user query using the fitted TF-IDF vectorizer
    query_vector = tfidf.transform([user_query])
    
    # Compute cosine similarity with all recipes
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    
    # Get top N indices
    top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
    
    # Create results dataframe
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = cosine_similarities[top_indices]
    
    # Filter out zero scores (no match at all)
    results = results[results['similarity_score'] > 0]
    for idx, (_, row) in enumerate(results.iterrows(), 1):
        print(f"{idx}. {row['title']}")
        print(f"   Score: {row['similarity_score']:.3f}")
        
        print(f"   Ingredients: {row['NER_flat'][:80]}...")
        if 'link' in df.columns and pd.notna(row['link']):
            print(f"   Link: {row['link']}")
        print()
    return results

# ========================
# 4. USAGE EXAMPLES
# ========================

# Example 1: Simple ingredient search
print("\n=== Search: 'chicken cream cheese' ===")
results = search_recipes_tfidf("chicken cream cheese", top_n=5)
print(results[['title', 'similarity_score']])

# Example 2: Multiple ingredients
print("\n=== Search: 'pasta mushrooms garlic' ===")
results = search_recipes_tfidf("pasta mushrooms garlic", top_n=5)
print(results[['title', 'similarity_score']])

# Example 3: Recipe name + ingredients
print("\n=== Search: 'creamy soup chicken' ===")
results = search_recipes_tfidf("creamy soup chicken", top_n=5)
print(results[['title', 'similarity_score']])

# Example 4: Check if no results
print("\n=== Search: 'unicorn magic' ===")
results = search_recipes_tfidf("unicorn magic", top_n=5)
if results.empty:
    print("No matching recipes found!")
else:
    print(results[['title', 'similarity_score']])