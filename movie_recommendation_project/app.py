from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import matplotlib.pyplot as plt

app = Flask(__name__)

ratings = pd.read_csv("ratings.csv").head(30000)
movies = pd.read_csv("movies.csv")

# Encode IDs
user_ids = ratings.userId.unique().tolist()
movie_ids = ratings.movieId.unique().tolist()

user_map = {x:i for i,x in enumerate(user_ids)}
movie_map = {x:i for i,x in enumerate(movie_ids)}

ratings["user"] = ratings["userId"].map(user_map)
ratings["movie"] = ratings["movieId"].map(movie_map)

num_users = len(user_ids)
num_movies = len(movie_ids)

X = ratings[["user","movie"]].values
y = ratings["rating"].values

# Deep learning model
user_input = tf.keras.layers.Input(shape=(1,))
movie_input = tf.keras.layers.Input(shape=(1,))

user_embed = tf.keras.layers.Embedding(num_users,50)(user_input)
movie_embed = tf.keras.layers.Embedding(num_movies,50)(movie_input)

u = tf.keras.layers.Flatten()(user_embed)
m = tf.keras.layers.Flatten()(movie_embed)

x = tf.keras.layers.Concatenate()([u,m])
x = tf.keras.layers.Dense(128,activation="relu")(x)
x = tf.keras.layers.Dense(64,activation="relu")(x)

output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model([user_input,movie_input],output)

model.compile(loss="mse",optimizer="adam",metrics=["mae"])

history = model.fit(
    [X[:,0],X[:,1]],
    y,
    epochs=2,
    batch_size=64
)
model.save("recommender_model.h5")
# Save training accuracy graph
plt.plot(history.history["mae"])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.savefig("static/chart.png")


# Get poster from OMDb
def get_poster(title):

    url = f"http://www.omdbapi.com/?t={title}&apikey=demo"
    data = requests.get(url).json()

    if data.get("Poster") and data["Poster"] != "N/A":
        return data["Poster"]

    return ""


# Recommendation function
def recommend(movie_title):

    movie = movies[movies.title.str.contains(movie_title,case=False)]

    if movie.empty:
        return []

    preds=[]

    for m in range(num_movies):

        p = model.predict([np.array([0]),np.array([m])],verbose=0)
        preds.append((m,p[0][0]))

    preds = sorted(preds,key=lambda x:x[1],reverse=True)[:10]

    results=[]

    for i,_ in preds:

        title = movies[movies.movieId==movie_ids[i]].title.values[0]
        poster = get_poster(title)

        results.append({
            "title":title,
            "poster":poster
        })

    return results


@app.route("/",methods=["GET","POST"])
def home():

    recs=[]

    if request.method=="POST":
        movie=request.form["movie"]
        recs=recommend(movie)

    return render_template(
        "index.html",
        movies=list(movies.title.values)[:200],
        recs=recs
    )

if __name__ == "__main__":
    app.run(debug=True)