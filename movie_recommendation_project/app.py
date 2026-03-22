from sklearn.metrics.pairwise import cosine_similarity

# create pivot table
movie_user_matrix = ratings.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)

similarity_matrix = cosine_similarity(movie_user_matrix)

movie_index_map = {movie_id: idx for idx, movie_id in enumerate(movie_user_matrix.index)}

def recommend(movie_name):
    movie_name = movie_name.lower()

    match = movies[movies["title"].str.lower().str.contains(movie_name, na=False)]

    if match.empty:
        sample = movies.sample(10)
        return [{"title": row.title, "poster": get_poster(row.title)} for _, row in sample.iterrows()]

    movie_id = match.iloc[0]["movieId"]

    if movie_id not in movie_index_map:
        return []

    idx = movie_index_map[movie_id]

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

    results = []

    for i, _ in similarity_scores:
        mid = movie_user_matrix.index[i]
        title = movies[movies.movieId == mid].title.values[0]

        results.append({
            "title": title,
            "poster": get_poster(title)
        })

    return results