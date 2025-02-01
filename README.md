# DocuChat - Open Source Document Assistant

## Project Setup Guide

### Prerequisites

Ensure you have the following installed on your machine:

- Python (>= 3.8)
- pip (latest version)
- Tesseract OCR (for image-based text extraction)
- Google Generative AI API Key
- Groq API Key

### Installation

Follow these steps to set up the project on a new machine:

#### 1. Clone the Repository

```sh
git clone <repository_url>
cd <repository_folder>
```

#### 2. Create and Activate Virtual Environment

```sh
python -m venv venv  # Create virtual environment
source venv/bin/activate  # Activate on MacOS/Linux
venv\Scripts\activate  # Activate on Windows
```

#### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

#### 4. Set Up Environment Variables

Create a `.env` file in the project root and add the following:

```
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

#### 5. Install and Configure Tesseract OCR

- **Windows**: Download and install Tesseract from [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux/macOS**:
  ```sh
  sudo apt install tesseract-ocr  # Debian-based
  brew install tesseract  # macOS
  ```
- Add Tesseract path to your environment variables (Windows example):
  ```py
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

### Running the Project

Start the Streamlit application:

```sh
streamlit run app.py
```

### Dependencies

The following Python packages are required:

```sh
streamlit
pytesseract
python-dotenv
PyPDF2
langchain
langchain_google_genai
FAISS
pypdfium2
Pillow
langchain_groq
```

If any dependency is missing, install it manually using:

```sh
pip install <package_name>
```

### Troubleshooting

- If Tesseract OCR is not working, ensure the correct path is set.
- If API keys are invalid, verify them in `.env`.
- If PDFs fail to process, check their content type and size.

![Alt text](C:\Users\HARSH JIVANI\OneDrive\Desktop\ChatPDF\2.jpg "Title")

### Contribution

Feel free to reach out to us if you have any questions or suggestions.
