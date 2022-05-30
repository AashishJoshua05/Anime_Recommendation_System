#-------------------------------------------------------------------------------------------------------
"""
This recommendation system uses the rating prediction to suggest similiar anime
to watch for the user, it makes use of the kaggle anime data set = 
https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
and inspiration was taken from GeeksforGeeks website = 
https://www.geeksforgeeks.org/recommendation-system-in-python/

A vector Rx is taken for the user's anime watched, and it is being compared to 
other users rating of that anime and being recommended similiar anime to which
the other users liked and rated in a similiar manner.
The formula is stored in the Screenshot in this folder.

"""
#-------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from time import sleep
from os import system
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

anime_df = pd.read_csv('anime.csv')
rating_df = pd.read_csv('rating.csv')


#-------------------------------------------------------------------------------------------------------
def CreateSparseMatrix(rating_df, anime_df):
    N = len(rating_df['user_id'].unique()) # 73515 No. Of users in data
    M = len(rating_df['anime_id'].unique()) # 11200 No. of anime in data

    user_mapper = dict(zip(np.unique(rating_df["user_id"]), list(range(N)))) #Mapping each used_id to a number in a sorted order in a dictionary
    anime_id_to_number = dict(zip(np.unique(rating_df["anime_id"]), list(range(M))))#Mapping each anime_id to a number in a sorted order in a dictionary 

    number_to_anime_id = dict(zip(list(range(M)), np.unique(rating_df["anime_id"])))#Mapping each number to a anime_id in a sorted order in a dictionary

    user_index = [user_mapper[i] for i in rating_df['user_id']] #Row for sparse matridx
    anime_index = [anime_id_to_number[i] for i in rating_df['anime_id']] #Column for sparse matrix

    X = csr_matrix((rating_df["rating"], (anime_index, user_index)), shape=(M, N))#Where anime_index = row, user_index = column and rating_df = the data
    anime_titles = dict(zip(anime_df['anime_id'], anime_df['name']))#Dict containts anime id as key and name as value
    
    return X, anime_id_to_number, number_to_anime_id, anime_titles
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
def SuggestSimilarAnime(X, anime_id_to_number, number_to_anime_id, watched_anime_id):
    # anime_id = 5114 #5114 #Anime_id for Full Metal Alchemist
    k=11 # Top n suggesting anime
    neighbour_ids = [] # List to store recommended anime ids
    try:
        anime_ind = anime_id_to_number[watched_anime_id]#Storing the Value of the anime_id key in dict which can be used in number_to_anime_id
    except:
        print("The anime has not yet ended wait for weekly episodes\n")
        return 0
    anime_vec = X[anime_ind] # Gets the ratings for the user watched anime given by 73k users
    kNN = NearestNeighbors(n_neighbors=k, algorithm="auto", metric='cosine') #Making K Nearest neighbors object
    kNN.fit(X) #Fitting the ratings of each anime in the KNN object
    neighbour = kNN.kneighbors(anime_vec, return_distance=False) # Stores an array of K nearest neighbors.

    for i in range(0,11):
        n = neighbour.item(i)
        neighbour_ids.append(number_to_anime_id[n])
    #Stores the ids of the neighbors according to the data. as neighbors contains the number not id
    neighbour_ids.pop(0)
    return neighbour_ids
#-------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------
def GetIDs(anime_watched, anime_titles):
    anime_titles_lower = dict((k, v.lower()) for k, v in anime_titles.items())
    key_list = list(anime_titles_lower.keys())
    val_list = list(anime_titles_lower.values())
    try:
        anime_watched_id = key_list[val_list.index(anime_watched)]
        return anime_watched_id
    except:
        return 0
#-------------------------------------------------------------------------------------------------------



#Main
#-------------------------------------------------------------------------------------------------------
X, anime_id_to_number, number_to_anime_id, anime_titles = CreateSparseMatrix(rating_df, anime_df)

while True:
    anime_watched = input("Enter an anime that you have watched and liked\n->").lower()
    anime_watched_id = GetIDs(anime_watched, anime_titles)
    if anime_watched_id == 0:
        print("The anime name is wrong or doesn't exist\n try again.")
        sleep(2)
        system('cls')
        continue
    else:
        similarAnime = SuggestSimilarAnime(X, anime_id_to_number, number_to_anime_id, anime_watched_id)
        if similarAnime == 0:
            pass
        else:    
            print(f"Since you watched {anime_titles[anime_watched_id]} we would suggest watching\n")
            for i, j in enumerate(similarAnime, start=1):
                print(f"{i}){anime_titles[j]}")
            sleep(5)
        print("Did you watch another anime?\n")
        print("Type 1 for Yes\nType 2 for No\n")
        yesno = int(input("->"))
        if yesno == 1:
            system('cls')
            continue
        else:
            print("Thank you bye :)")
            sleep(2)
            system('cls')
            break
#-------------------------------------------------------------------------------------------------------
