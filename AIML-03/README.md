# Large-Scale Personalized Recommendation Engine

## About the Project

This project is a personalized recommendation system developed as part of my Nexvia Technology internship.

The main goal of this project is to recommend movies to users based on their interests and previous ratings. It uses a Two-Tower Neural Network to generate user and movie embeddings, FAISS for finding similar movies, and a ranking model to improve the final recommendations.

I also integrated Redis for caching, Apache Spark for data processing, Apache Airflow for pipeline validation, and Triton Inference Server for model inference.

## Technologies Used

- **Python** – Main programming language
- **NumPy, Pandas** – Data handling and preprocessing
- **Scikit-learn** – Data preprocessing and evaluation
- **TensorFlow** – Building and training deep learning models
- **TensorFlow Recommenders (TFRS)** – Two-Tower recommendation model
- **FAISS** – Fast similarity search and candidate retrieval
- **Redis** – Caching recommendations
- **Apache Spark** – Data processing
- **Apache Airflow** – Pipeline validation and task management
- **Triton Inference Server** – Model inference
- **ONNX** – Model export and inference compatibility
- **Jupyter Notebook** – Development and experimentation
- **Docker** – Running infrastructure services
- **Matplotlib, Seaborn** – Data visualization

## Project Features

1. **Data Exploration:** Analyzes user ratings, movie information, genres, and rating patterns.
2. **Data Preprocessing:** Cleans and prepares the MovieLens dataset for model training.
3. **Two-Tower Model:** Generates 64-dimensional user and movie embeddings.
4. **Candidate Retrieval:** Uses FAISS to find relevant movies for each user.
5. **Ranking Model:** Uses a neural network to rank the retrieved movie recommendations.
6. **Model Evaluation:** Evaluates recommendations using Recall@10, Precision@10, and NDCG@10.
7. **Recommendation Pipeline:** Generates personalized top-10 movie recommendations for users.
8. **Redis Caching:** Stores recommendations to reduce repeated processing.
9. **Triton Integration:** Serves the ranking model using Triton Inference Server.
10. **Airflow Integration:** Validates model artifacts and recommendation data through an Airflow DAG.

## How the System Works

The system has different components that work together:

1. The MovieLens dataset is loaded and explored.
2. The data is cleaned and prepared for training.
3. The Two-Tower model generates user and movie embeddings.
4. FAISS searches for relevant movie candidates using the embeddings.
5. The ranking model scores and sorts the retrieved movies.
6. The system selects the top 10 recommendations for each user.
7. Redis stores recommendations for faster future retrieval.
8. Triton handles ranking model inference.
9. Airflow runs validation tasks for the pipeline.

## Dataset

This project uses the **MovieLens 1M dataset** provided by GroupLens.

The dataset contains movie ratings and user information.

| Dataset Information | Count |
|---|---:|
| Users | 6,040 |
| Movies | 3,883 |
| Ratings | 1,000,209 |
| Genres | 18 |

The dataset contains three main files:

- `ratings.dat` – User ratings for movies
- `movies.dat` – Movie titles and genres
- `users.dat` – User demographic information

Ratings of 4 and 5 were treated as positive interactions for the recommendation model.

Dataset source: [MovieLens 1M – GroupLens](https://grouplens.org/datasets/movielens/1m/)

## Project Structure

```text
Large-Scale Personalized Recommendation Engine/
├── dataset/
├── demo/
├── Models/
├── notebooks/
├── Outputs/
└── Documentation/
```

## Notebooks

The project is divided into eight Jupyter notebooks:

| Notebook | Purpose |
|---|---|
| `01_Data_Exploration.ipynb` | Dataset exploration and analysis |
| `02_Data_Preprocessing.ipynb` | Data cleaning and feature preparation |
| `03_Two_Tower_Model.ipynb` | Training the Two-Tower model |
| `04_Candidate_Retrieval_FAISS.ipynb` | Finding movie candidates using FAISS |
| `05_Ranking_Model.ipynb` | Training the recommendation ranking model |
| `06_Model_Evaluation.ipynb` | Evaluating recommendation performance |
| `07_Recommendation_Pipeline.ipynb` | Generating personalized recommendations |
| `08_Infrastructure_Integration.ipynb` | Integrating Redis, Spark, Airflow, and Triton |

## Model Performance and Results

The recommendation system was evaluated using a chronological train-test split.

The following results were obtained:

| Metric | FAISS Retrieval | Ranking Model |
|---|---:|---:|
| Recall@10 | 0.0565 | 0.0565 |
| Precision@10 | 0.0612 | 0.0612 |
| NDCG@10 | 0.0736 | 0.0765 |

The ranking model improved NDCG@10 by approximately **3.94%** compared to the initial FAISS retrieval results.

Other results:

- Ranking model validation AUC: **0.8040**
- TensorFlow and Triton inference difference: **5.96 × 10⁻⁸**
- Redis cache-hit response time: approximately **0.0066–0.0179 seconds**

The Redis timings are from cache-hit tests, not uncached recommendation generation.

## Project Setup

### 1. Clone or Open the Project

Open the project folder in VS Code or Anaconda Prompt.

### 2. Create the Conda Environment

Create and activate the environment:

```bash
conda create -n recommendation_env python=3.10
conda activate recommendation_env
```

Set the TensorFlow compatibility variable:

```bash
set TF_USE_LEGACY_KERAS=1
```

Install the required project dependencies in the environment before running the notebooks.

### 3. Start Jupyter Notebook

Run:

```bash
jupyter notebook
```

Open the `notebooks` folder and run the notebooks in order, starting from data exploration.

### 4. Infrastructure Setup

Docker Desktop is required for running the infrastructure services used in this project.

Redis, Airflow, and Triton have separate configurations. Their required model files, data, and service configurations must be available before running the related integration notebook.

## Testing and Validation

I tested the recommendation pipeline and its infrastructure components.

The following features were tested successfully:

- Two-Tower model training completed successfully.
- FAISS generated movie candidates for users.
- Ranking model generated recommendation scores.
- Top-10 recommendations were generated for supported users.
- Evaluation metrics were calculated successfully.
- Redis stored and retrieved cached recommendations.
- Spark ran successfully in local mode.
- Airflow validation tasks completed successfully.
- Triton loaded the ranking model successfully.
- TensorFlow and Triton inference outputs were compared successfully.

These tests confirm that the implemented components work together, but they do not guarantee that every recommendation will match a user's actual preferences.

## Project Outputs

- Trained user and movie embedding models
- FAISS movie index
- Trained ranking model
- Personalized top-10 movie recommendations
- Evaluation metrics and comparison graphs
- Processed dataset files
- Recommendation output CSV files
- Redis cache results
- Triton inference results
- Airflow pipeline validation status

## Future Improvements

- Improve recommendation quality with more advanced models.
- Add more user and movie features.
- Improve cold-start recommendations for new users and movies.
- Test the system on larger datasets.
- Improve recommendation response time.
- Add a web-based interface for users.
- Deploy the complete recommendation system.

## Note

This project was developed for learning and research purposes as part of my Nexvia Technology internship. The system was tested locally using CPU-based infrastructure and has not been deployed as a public recommendation service.

---

**Developed as part of my Nexvia Technology Internship.**