## cv-matcher: AI-Powered Resume Matching System


cv-matcher is an intelligent applicant tracking system that uses AI to match resumes with job descriptions. It provides a similarity score and suggests the best-fitting job roles based on resume content.

---

## Key Features

- **AI-Powered Matching**: Semantic analysis using Sentence Transformers
- **Enhanced AI Matching**: Powered by Google Gemini for detailed resume and job description analysis
- **Multi-Step Workflow**: Intuitive 3-step process for resume and job analysis
- **File & Text Support**: Upload PDF/DOCX files or paste text directly
- **Visual Results**: Clear similarity score visualization with color coding
- **Top Job Suggestions**: Identifies best-fitting roles with similarity percentages
- **Side-by-Side Comparison**: View resume and job description together
- **Privacy Focused**: Automatic file cleanup after processing
- **DOCX Report Generation**: Download a detailed analysis report of your resume and job description match

---

## Technology Stack

| Component           | Technology                          |
| ------------------- | ----------------------------------- |
| **Frontend**        | HTML5, CSS3, JavaScript             |
| **Backend**         | Python, Flask                       |
| **AI Engine**       | Sentence Transformers, Google Gemini |
| **Text Processing** | NLTK, PyPDF2, python-docx           |
| **Deployment**      | Docker (optional)                   |

---

## Website Interface

### Step 1: Resume Upload


- Upload PDF/DOCX resume or paste text
- Clean, modern interface with step indicators

### Step 2: Job Description Upload


- Upload job description or paste text
- Visual progress tracking

### Results Page


- Similarity score visualization
- Top job matches with similarity percentages
- Side-by-side resume and JD comparison
- Detailed analysis from Google Gemini
- **Download Report**: Generate and download a comprehensive DOCX report

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager
- Google Gemini API Key (See below)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/nicounii/cv-matcher.git
   cd cv-matcher
   ```

2. **Create virtual environment**:

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If you encounter issues with `pandas`, try installing it separately:
   ```bash
   pip install pandas==2.2.0
   ```

4. **Download NLTK data**:

   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

5. **Configure API Key**:

   Create a `.env` file in the project root and add your Google Gemini API Key:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

   You can obtain a key from [Google AI Studio](https://aistudio.google.com/).

### Running the Application

```bash
python app.py
```

Access the application at: `http://localhost:5000`

### Docker Setup

```bash
docker build -t cv-matcher .
docker run -p 5000:5000 cv-matcher
```

---

## Usage

1. **Upload Resume**:

   - Select a PDF/DOCX file or paste resume text
   - Click **Continue**

2. **Upload Job Description**:

   - Select a job description file or paste text
   - Click **Analyze**

3. **View Results**:

   - See your match percentage
   - Explore top job suggestions
   - Compare resume and job description
   - **Download Report**: Click to download a detailed DOCX report
   - Click **Analyze More** to start over

---

## Data Sources & Models

### AI Models

- **Core Matching Model**: [bwbayu/sbert\_model\_jobcv](https://huggingface.co/bwbayu/sbert_model_jobcv)
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Analysis Model**: Google Gemini (1.5 Flash/Pro)

### Training Data Sources

- [bwbayu/job\_cv\_supervised](https://huggingface.co/datasets/bwbayu/job_cv_supervised)
- [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
- [InferencePrince555/Resume-Dataset](https://huggingface.co/datasets/InferencePrince555/Resume-Dataset)
- [Jog Description and Salary in Indonesia](https://www.kaggle.com/datasets/canggih/jog-description-and-salary-in-indonesia)
- [IT Job Post Descriptions](https://www.kaggle.com/datasets/mscgeorges/itjobpostdescriptions)
- [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

### Job Taxonomy

- System Administrator
- Database Administrator
- Web Developer
- Security Analyst
- Network Administrator
- Data Scientist
- DevOps Engineer
- Cloud Engineer
- Machine Learning Engineer
- Software Engineer

---

**License**: MIT License\
**Contact**: [geo.matheussantos@gmail.com](mailto\:geo.matheussantos@gmail.com)\
**Version**: 1.0.0

