import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def load_data():
    df = pd.read_csv('amazon.csv')
    df['rating_count'] = pd.to_numeric(df['rating_count'].replace({',': ''}, regex=True), errors='coerce')
    df['rating_count'] = df['rating_count'].fillna(df['rating_count'].median()).astype(int)
    df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
    df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
    df['discount_percentage'] = df['discount_percentage'].replace({'%': '', ',': ''}, regex=True).astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['category'] = df['category'].astype(str).apply(lambda x: x.split('|'))
    return df

def compute_similarity(df):
    mlb = MultiLabelBinarizer()
    category_binarized = mlb.fit_transform(df['category'])
    category_df = pd.DataFrame(category_binarized, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, category_df], axis=1).drop('category', axis=1)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    df['product_name_tfidf'] = list(tfidf_vectorizer.fit_transform(df['product_name']).toarray())
    df['about_product_tfidf'] = list(tfidf_vectorizer.fit_transform(df['about_product']).toarray())
    
    product_name_tfidf = np.array(df['product_name_tfidf'].tolist())
    about_product_tfidf = np.array(df['about_product_tfidf'].tolist())
    
    cosine_sim = 0.6 * cosine_similarity(product_name_tfidf) + 0.4 * cosine_similarity(about_product_tfidf)
    category_sim = 1 - cdist(csr_matrix(category_binarized).toarray(), csr_matrix(category_binarized).toarray(), metric='jaccard')
    
    return cosine_sim, category_sim

def get_recommendations(product_name, final_sim, df, top_n=5):
    if product_name not in df['product_name'].values:
        return "Product name not found."
    product_idx = df[df['product_name'] == product_name].index[0]
    sim_scores = sorted(list(enumerate(final_sim[product_idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return df.iloc[[i[0] for i in sim_scores]][['product_name', 'discounted_price', 'rating']]

def train_svd_model(df):
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=50, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    return model, rmse

def recommend_products(user_id, model, df, n=5):
    all_products = df['product_id'].unique()
    user_rated_products = df[df['user_id'] == user_id]['product_id'].tolist()
    
    predictions = [model.predict(user_id, pid) for pid in all_products if pid not in user_rated_products]
    top_products = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    recommended_products = df[df['product_id'].isin([p.iid for p in top_products])][['product_name', 'discounted_price', 'rating']]
    
    return recommended_products

def get_hybrid_recommendations(user_id, product_name, model, final_sim, df, top_n=5):
    if product_name not in df['product_name'].values:
        return "Product not found."

    product_idx = df[df['product_name'] == product_name].index[0]

    sim_scores = sorted(list(enumerate(final_sim[product_idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    content_recommendations = df.iloc[[i[0] for i in sim_scores]][['product_name', 'discounted_price', 'rating']]

    all_products = df['product_id'].unique()
    user_rated_products = df[df['user_id'] == user_id]['product_id'].tolist()
    predictions = [model.predict(user_id, pid) for pid in all_products if pid not in user_rated_products]
    top_cf_products = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]

    cf_recommendations = df[df['product_id'].isin([p.iid for p in top_cf_products])][['product_name', 'discounted_price', 'rating']]

    hybrid_recommendations = pd.concat([content_recommendations, cf_recommendations]).drop_duplicates().head(top_n)
    
    return hybrid_recommendations


st.title("Amazon Product Recommendation System")

if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df
cosine_sim, category_sim = compute_similarity(df)
final_sim = 0.5 * cosine_sim + 0.5 * category_sim

st.sidebar.header("Product Recommendation")
product_name = st.sidebar.selectbox("Select Product Name:", df['product_name'].unique())
if st.sidebar.button("Get Recommendations"):
    recommendations = get_recommendations(product_name, final_sim, df)
    st.write(recommendations)

st.sidebar.header("User-Based Recommendation")
user_id = st.sidebar.text_input("Enter User ID:")
if st.sidebar.button("Recommend for User"):
    model, _ = train_svd_model(df)  # No need to return RMSE
    if model:
        recommendations = recommend_products(user_id, model, df)
        st.write(recommendations)
    else:
        st.write("Not enough rating data to generate recommendations.")

st.sidebar.header("Hybrid Recommendation System")

if st.sidebar.button("Get Hybrid Recommendations"):
    model, _ = train_svd_model(df)  # Train CF model
    if model:
        hybrid_recommendations = get_hybrid_recommendations(user_id, product_name, model, final_sim, df)
        st.write(hybrid_recommendations)
    else:
        st.write("Not enough rating data to generate recommendations.")

