from datasets import load_dataset
import pandas as pd
import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
import logging
from ..core.config import DATASETS, SKILLS_DB_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return list(set(tokens))


def load_all_datasets():
    job_taxonomy = []
    skills_db = set()

    # Load Hugging Face datasets
    for ds_name in DATASETS:
        if "/" in ds_name:  # Only load HF datasets that have the format "user/dataset"
            try:
                logger.info(f"Loading Hugging Face dataset: {ds_name}")
                dataset = load_dataset(ds_name)

                if "train" in dataset:
                    df = pd.DataFrame(dataset["train"])
                else:
                    df = pd.DataFrame(dataset)

                # Extract job titles and descriptions
                if "job_title" in df.columns and "job_description" in df.columns:
                    for _, row in df.iterrows():
                        job_taxonomy.append(
                            f"{row['job_title']} - {clean_text(row['job_description'])}"
                        )
                        skills_db.update(extract_skills(row["job_description"]))

                # Extract skills from resume datasets
                if "resume" in df.columns:
                    for resume in df["resume"]:
                        skills_db.update(extract_skills(clean_text(resume)))

                logger.info(f"Loaded {len(df)} records from {ds_name}")
            except Exception as e:
                logger.error(f"Error loading {ds_name}: {e}")

    # Load Kaggle datasets from CSV files
    kaggle_path = "data/kaggle"
    if os.path.exists(kaggle_path):
        # Process all CSV files in the kaggle directory
        for file in os.listdir(kaggle_path):
            if file.endswith(".csv"):
                try:
                    file_path = os.path.join(kaggle_path, file)
                    logger.info(f"Loading Kaggle dataset: {file}")
                    df = pd.read_csv(file_path)

                    # Extract based on common column names
                    title_col = None
                    desc_col = None
                    skills_col = None

                    for col in df.columns:
                        col_lower = col.lower()

                        if "title" in col_lower and "job" in col_lower:
                            title_col = col
                        if "desc" in col_lower or "description" in col_lower:
                            desc_col = col
                        if "skills" in col_lower:
                            skills_col = col

                    # Process job titles and descriptions
                    if title_col and desc_col:
                        for _, row in df.iterrows():
                            job_taxonomy.append(
                                f"{row[title_col]} - {clean_text(str(row[desc_col]))}"
                            )

                    # Process skills
                    if skills_col:
                        for skills in df[skills_col]:
                            if isinstance(skills, str):
                                skills_db.update(
                                    [s.strip().lower() for s in skills.split(",")]
                                )

                    logger.info(f"Loaded {len(df)} records from {file}")
                except Exception as e:
                    logger.error(f"Error loading Kaggle dataset {file}: {e}")

    # Deduplicate and save
    job_taxonomy = list(set(job_taxonomy))
    skills_db = list(skills_db)

    # Save skills database
    joblib.dump(skills_db, SKILLS_DB_PATH)
    logger.info(f"Saved skills database with {len(skills_db)} skills")

    # Fallback if no data loaded
    if not job_taxonomy:
        logger.warning("No job taxonomy loaded, using default taxonomy")
        job_taxonomy = [
            "System Administrator - Manages computer systems and networks",
            "Database Administrator - Designs, implements and maintains database systems",
            "Web Developer - Builds and maintains websites and web applications",
            "Security Analyst - Protects computer systems and networks from cyber threats",
        ]

    return pd.DataFrame(job_taxonomy, columns=["job_title"])
