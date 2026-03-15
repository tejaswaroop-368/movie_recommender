# 🎬 Movie Recommendation System (NNDL Project)

A full-stack **Movie Recommendation System** built using **Deep Learning and Flask**.
The system recommends movies based on user preferences using a neural network trained on the MovieLens dataset.

---

## 🚀 Features

* 🔍 Movie search functionality
* 🎥 Automatic movie posters using OMDb API
* 🤖 Deep Learning recommendation model
* 📊 Training accuracy visualization
* 🌙 Netflix-style dark UI
* 🌐 Flask web application

---

## 🧠 Model Architecture

The recommendation model uses **Neural Collaborative Filtering**.

User ID → Embedding Layer
Movie ID → Embedding Layer
↓
Concatenation
↓
Dense Layers (ReLU)
↓
Predicted Rating

The model predicts how much a user would like a movie and recommends the top results.

---

## 📂 Project Structure

```
movie_recommendation_project
│
├── app.py
├── recommender_model.keras
├── movies.csv
├── ratings.csv
│
├── templates
│   └── index.html
│
├── static
│   ├── style.css
│   └── chart.png
```

---

## 🛠 Technologies Used

Backend:

* Python
* Flask
* TensorFlow
* Pandas
* NumPy

Frontend:

* HTML
* CSS

Dataset:

* MovieLens Dataset

API:

* OMDb API

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/vamsi4956/movie_recommendation_project.git
```

Navigate to the project folder:

```
cd movie_recommendation_project
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 Dataset

This project uses the **MovieLens dataset**, which contains user movie ratings used to train recommendation algorithms.

---

## 🎯 Future Improvements

* Personalized user accounts
* Collaborative filtering improvements
* Movie trailers integration
* Real-time search suggestions
* Netflix-style UI enhancements

---

## 👨‍💻 Author

Vamsi
NNDL Project – 2026
