#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')


# In[3]:


movies


# # Cleaning titles for the engine

# In[4]:


def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# In[5]:


movies["clean_title"] = movies["title"].apply(clean_title)


# In[6]:


movies


# # Creating tfidf matrix

# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer =  TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])


# # Search function
# 

# In[8]:


from sklearn.metrics.pairwise import cosine_similarity
#import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()  #compare the queries and return similarity
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    return results


# In[9]:


#testing
title = "Skinford: Death Sentence"
title = clean_title(title)
query_vec = vectorizer.transform([title])
similarity = cosine_similarity(query_vec, tfidf).flatten()
indices = np.argpartition(similarity, -5)[-5:]
results = movies.iloc[indices].iloc[::-1]
results


# # Search Box

# In[10]:


#pip install ipywidgets


# In[11]:


# Create the input text box and output widget
movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_output = widgets.Output()


def on_type(data):
    with movie_output:
        clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_output.clear_output()  
            display(results)


movie_input.observe(on_type, names='value')
display(movie_input, movie_output)


# In[12]:


# Define the movie_id you want to find similar users for
movie_id = 1  # For example, Toy Story

# Find similar users
similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
similar_user_recs = similar_user_recs.value_counts() / len(similar_users)


# In[13]:


all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
rec_percentages.columns = ["similar", "all"]
rec_percentages


# In[14]:


rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
rec_percentages = rec_percentages.sort_values("score", ascending=False)
rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")


# In[15]:


def find_similar_movies(movie_id):
    similar_movies = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_movies)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_movies)
    return similar_user_recs


# In[16]:


movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()


# # Recommendation Engine

# In[25]:


movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            # Calculate a dummy score based on 'movieId' and 'title'
            results["score"] = results["movieId"] % 100  # Just an example calculation
            display(results[['score', 'movieId', 'title', 'genres']])

movie_input.observe(on_type, names='value')

display(movie_input, recommendation_list)




# In[ ]:




