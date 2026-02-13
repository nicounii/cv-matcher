# Configuration for datasets and models
DATASETS = [
    # Hugging Face datasets
    "bwbayu/job_cv_supervised",
    "cnamuangtoun/resume-job-description-fit",
    "InferencePrince555/Resume-Dataset",
    # Kaggle datasets (manually downloaded and placed in data/kaggle/)
    "jog-description-and-salary-in-indonesia",
    "itjobpostdescriptions",
    "resume-dataset",
]

MODEL_NAME = "bwbayu/sbert_model_jobcv"
EMBEDDINGS_PATH = "data/models/job_embeddings.pkl"
SKILLS_DB_PATH = "data/cache/skills_db.pkl"
