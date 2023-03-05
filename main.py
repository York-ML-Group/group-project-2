import pandas as pd
import streamlit as st
import pickle

dataset = pd.read_csv("./dataset.csv")

artists = dataset['artists'].unique()
file_name = './kmeans_model.pickle'

loaded_model = pickle.load(open(file_name, "rb"))

dataset['cluster'] = loaded_model.labels_
# print(dataset.head(4))


def get_tracks(artist):
    filtered_song = dataset['artists'] == artist
    return dataset[filtered_song]['track_name'].unique()


def get_recommended_playlist(input_song):
    cluster_id = dataset['track_name'] == input_song
    # print(input_song)
    selected_cluster = dataset[cluster_id]['cluster'].iloc[0]
    recommendations_raw = dataset['cluster'] == selected_cluster
    unique_recommendations = dataset[recommendations_raw]
    final_suggestions = unique_recommendations.sample(n=10)
    # print(final_suggestions)
    return final_suggestions


with st.sidebar:
    st.header('Select an artist and a song to get a recommended playlist')
    artist_selected = st.selectbox('Select an artist', options=artists, key=1)
    song_selected = st.selectbox('Select a song', options=get_tracks(artist_selected), key=2)


user_song = song_selected
user_artist = artist_selected
st.write('Your selected artist is : {}'.format(user_artist))
st.write('Your selected song is : {}'.format(user_song))

st.subheader('Our recommended playlist based on your selections:')

song_suggestions = get_recommended_playlist(user_song)
song_suggestions.to_markdown(index=False)
st.write(song_suggestions[['artists', 'track_name']])
print('Run completed')
