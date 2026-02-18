import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from wordcloud import WordCloud
import faiss

DATA_URL = "./data/eCommerce_Item_Data.parquet"

### Config
st.set_page_config(
    page_title="E-commerce – Catalogue produits",
    page_icon="🌐",
    layout="wide"
)
### Data
@st.cache_data
def load_data():
    columns = {
        'id': 'id produit',
        'topic_name': 'famille',
        'product_name': 'nom du produit',
        'description': 'description',
        'clean_token': 'tokens',
        'dimension_1': 'dimension_1',
        'dimension_2': 'dimension_2'
    }

    product_data = pd.read_parquet(
        DATA_URL,
        columns=list(columns.keys())
    ).rename(columns=columns)

    return product_data

@st.cache_data
def load_stopwords():
    with open("./data/stop_words.pkl", "rb") as file:
        stop_words = pickle.load(file)
    
    return stop_words

### Load model & index
@st.cache_resource
def load_index():
    index = faiss.read_index("./data/index.faiss")
    return index

@st.cache_data
def load_embeddings():
    return np.load("./data/embeddings.npy")

product_data = load_data()
stop_words = load_stopwords()
index = load_index()
embeddings = load_embeddings()

### Streamlit pages

### Title page
def title_page():
    st.title("E-commerce – Catalogue de ventes en ligne")
    st.subheader("Catalogue des produits")

    col1, col2 = st.columns(2)
    with col1:
        product_id_search = st.text_input("Rechercher un produit par son ID")

    with col2:
        product_family_search = st.selectbox(
            "Rechercher un produit par sa famille",
            [""] + list(product_data["famille"].unique())
        )
    product_description_search = st.text_input("Rechercher un produit par sa description")

    mask = pd.Series(True, index=product_data.index)

    if product_id_search:
        mask &= product_data["id produit"] == int(product_id_search)
    if product_description_search:
        mask &= product_data["description"].str.contains(product_description_search, case=False, na=False)
    if product_family_search:
        mask &= product_data["famille"].str.contains(product_family_search, na=False)

    product_list = product_data[mask]

    st.write(f"Nombre de produits trouvés : {len(product_list)}")
    if not product_list.empty:
        st.dataframe(product_list[['id produit', 'famille', 'nom du produit', 'description']], 
                     hide_index=True, 
                     width='content')
    else:
        st.warning("Aucun produit trouvé avec ces critères.")

def topics_page():
    st.header("Clustering des sujets après réduction à deux dimensions")
    fig = px.scatter(
    x=product_data['dimension_1'],
    y=product_data['dimension_2'],
    color=product_data['famille'].astype("category"),
    title="Visualisation des clusters",
    labels={"x": "Dimension 1", "y": "Dimension 2"},
    hover_name=product_data['famille'],
    )
    fig.update_layout(legend_title="Familles de produits")
    st.plotly_chart(fig)
    

def word_topics_page():
    family_search = st.selectbox("Selectionner une famille de produits", product_data['famille'].unique())
    wordcloud_text = " ".join(
        [desc for desc in product_data['tokens']
        if product_data.loc[product_data['tokens'] == desc, 'famille'].values[0] == family_search]
    )

    wordcloud = WordCloud(
        background_color='white',
        max_words=10,
        stopwords=stop_words
    )

    wordcloud.generate(wordcloud_text)
    wordcloud_image = wordcloud.to_image()
    wordcloud_image = np.array(wordcloud_image)

    fig = go.Figure()

    fig.add_trace(
        go.Image(z=wordcloud_image)
    )

    st.plotly_chart(fig)

def recommandation_page():
    st.header("Liste des produits similaires")
    col1, col2 = st.columns(2)
    with col1:
        product_id = st.number_input("Renseigner l'ID du produit à remplacer", 
                                            min_value=1, 
                                            max_value=product_data['id produit'].max(),
                                            value=None)
    with col2:
        nb_products = st.number_input("Nombre de produits semblables souhaités",
                                    min_value=1, 
                                    max_value=10,
                                    value=5)

    if product_id is None:
        st.warning("Veuillez renseigner l'ID du produit à remplacer.")
    else:
        selected_product = product_data.loc[product_data['id produit'] == product_id]
        st.dataframe(selected_product[['id produit','nom du produit', 'description']], hide_index=True)

        query_embedding = embeddings[product_data['id produit'] == product_id].reshape(1, -1)
        distances, indices = index.search(query_embedding, nb_products)
        products = product_data.iloc[indices[0]][1:].reset_index(drop=True)

        similarities = 1 / (1 + distances)
        similarities_df = pd.DataFrame(similarities).T.iloc[1:].rename(columns={0:'similarity'}).reset_index(drop=True)

        results = pd.concat([products, similarities_df], axis=1)
        results['similarity'] = results['similarity'] * 100
        results.rename(columns={'id': 'id produit', 
                                'product_name': 'nom du produit', 
                                'description': 'description', 
                                'similarity': 'similitude'}, inplace=True)

        st.write("Liste des produits similaires:")
        st.dataframe(results[['id produit', 'nom du produit', 'description', 'similitude']].sort_values(by='similitude', ascending=False),
                     hide_index=True, 
                     width='stretch')

### Pages layout
pages = {
    "Catalogue produits": [
    st.Page(title_page, title="Catalogue de produits", icon="📒"),
    st.Page(recommandation_page, title="Trouver des produits similaires", icon="🤖"),
    ],
    "Informations familles produits": [
    st.Page(topics_page, title="Clustering des descriptions", icon="✨"),
    st.Page(word_topics_page, title="Nuages de mots par famille", icon="☁️"),
    ]
    }

pg = st.navigation(pages)

pg.run()