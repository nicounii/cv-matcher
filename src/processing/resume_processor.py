import re
from PyPDF2 import PdfReader
from docx import Document


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text


def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            if para.text is not None:
                text += para.text + "\n"
    except Exception as e:
        print(f"Error processing DOCX: {e}")
    return text


def normalize_display(text: str) -> str:
    """
    Preserve punctuation and special characters for UI display.
    Keep bullets, plus signs, dashes, ampersands, numbers, etc.
    Normalize whitespace but keep new lines.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")  # non‑breaking space
    # collapse spaces/tabs but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_for_model(text: str) -> str:
    """
    Create a normalized, low-noise copy for embeddings/ML matching only.
    This does NOT affect UI display.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)  # drop punctuation for model
    text = text.lower()
    # Keep numbers for ATS? For semantic matching we can drop them:
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_resume(file_path):
    # Returns display-safe text (unstripped), not cleaned
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
    return normalize_display(text)


def process_jd(file_path):
    return process_resume(file_path)
