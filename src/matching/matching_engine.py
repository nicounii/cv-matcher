from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined job taxonomy
JOB_TAXONOMY = [
    "System Administrator",
    "Database Administrator",
    "Web Developer",
    "Security Analyst",
    "Network Administrator",
    "Data Scientist",
    "DevOps Engineer",
    "Cloud Engineer",
    "Machine Learning Engineer",
    "Software Engineer",
]

# Model name
MODEL_NAME = "bwbayu/sbert_model_jobcv"
EMBEDDINGS_PATH = "data/models/job_embeddings.pkl"

# Initialize model
model = SentenceTransformer(MODEL_NAME)
EMBEDDING_DIM = model.get_sentence_embedding_dimension()

# Load or create job embeddings
job_embeddings = np.array([])
try:
    if os.path.exists(EMBEDDINGS_PATH):
        data = joblib.load(EMBEDDINGS_PATH)
        if "embeddings" in data and data["embeddings"].size > 0:
            job_embeddings = data["embeddings"]
            logger.info(f"Loaded job embeddings with shape: {job_embeddings.shape}")

    # Regenerate if empty or dimension mismatch
    if job_embeddings.size == 0:
        logger.info("Generating job embeddings...")
        embeddings = model.encode(JOB_TAXONOMY)
        job_embeddings = np.array(embeddings)
        joblib.dump(
            {"embeddings": job_embeddings, "dimension": EMBEDDING_DIM}, EMBEDDINGS_PATH
        )
        logger.info(f"Saved job embeddings with shape: {job_embeddings.shape}")

except Exception as e:
    logger.error(f"Error loading/generating job embeddings: {e}")
    # Fallback to generating on the fly
    job_embeddings = model.encode(JOB_TAXONOMY)


def calculate_similarity(resume_text, jd_text):
    try:
        # Generate embeddings
        resume_embedding = model.encode([resume_text])[0]
        jd_embedding = model.encode([jd_text])[0]

        # Calculate cosine similarity
        similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
        return round(similarity * 100, 2)
    except Exception as e:
        logger.error(f"Similarity calculation error: {e}")
        return 0.0


def get_top_job_matches(resume_text, top_n=5):
    try:
        # Generate resume embedding
        resume_embedding = model.encode([resume_text])[0]

        # Calculate similarity with all jobs
        similarities = cosine_similarity([resume_embedding], job_embeddings)[0]

        # Get top matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return [(JOB_TAXONOMY[i], round(similarities[i] * 100, 2)) for i in top_indices]

    except Exception as e:
        logger.error(f"Top job matches error: {e}")
        return []
