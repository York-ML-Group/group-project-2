import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

columns_to_cluster = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence', 'tempo',
                      'speechiness', 'loudness']

# Importing the mall dataset with pandas
df = pd.read_csv('./dataset.csv')
dftst = pd.read_csv('./testdata.csv')

# Using StandardScaler
ss = StandardScaler()
songs_scaled = ss.fit_transform(df[columns_to_cluster])
user_scaled = ss.fit_transform(dftst[columns_to_cluster])

columns_to_cluster_scaled = ['acousticness_scaled', 'danceability_scaled',
                             'energy_scaled', 'instrumentalness_scaled', 'liveness_scaled',
                             'valence_scaled', 'tempo_scaled', 'speechiness_scaled', 'loudness_scaled']

df_songs_scaled = pd.DataFrame(songs_scaled, columns=columns_to_cluster_scaled)

# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=6, max_iter=300, n_init=10, init='k-means++', random_state=42)
# fitting the k means algorithm on scaled data
kmeans.fit(songs_scaled)
# save model
filename = "C:\\python_workspace\\data_visualization\\music_albums_data_clustering\\kmeans_model.pickle"
pickle.dump(kmeans, open(filename, "wb"))

preds = kmeans.predict(user_scaled)
df_last_3 = df.tail(3)
last3_songs_scaled = ss.fit_transform(df_last_3[columns_to_cluster])

# load model
loaded_model = pickle.load(open(filename, "rb"))
print("K-Means model loaded ......")
pred_clusts = loaded_model.predict(last3_songs_scaled)
print(pred_clusts)

df_songs_scaled['cluster'] = loaded_model.labels_
print(df_songs_scaled.head(4))
df_songs_joined = pd.concat([df, df_songs_scaled], axis=1).set_index('cluster')
print(df_songs_scaled.head(4))

df['cluster'] = loaded_model.labels_
print(df.head(4))

out_csv_filename = "C:\\python_workspace\\data_visualization\\music_albums_data_clustering\\songs_with_cluster_index.csv"
df.to_csv(out_csv_filename)
