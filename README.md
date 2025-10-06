#Lab Report Digitization App

#Overview
This application is designed to digitize lab reports by extracting patient details and test results from uploaded PDF, JPG, PNG, or JPEG files. It uses OCR (Optical Character Recognition) with EasyOCR, natural language processing with spaCy, and rule-based extraction to process scanned or digital documents. The app provides a Streamlit-based UI and a FastAPI endpoint for programmatic access.
Features

#Upload Support: Accepts PDF, JPG, PNG, and JPEG files.
Preprocessing: Includes deskewing, denoising, and canvas expansion for better OCR results.
OCR Extraction: Uses EasyOCR to extract text and bounding box data from images.
Data Extraction: Combines rule-based and ML-based methods to identify patient details (e.g., name, age, gender, ID) and test results (e.g., Hemoglobin, Glucose).
Tabulated Output: Displays extracted data in editable tables via Streamlit.
Save and Download: Allows saving extracted data as JSON files and downloading them.
API Integration: Provides a FastAPI endpoint for automated processing.

#Prerequisites

Python 3.11+
Required Python packages:
streamlit
fastapi
uvicorn
easyocr
spacy
pandas
pillow
numpy
opencv-python
pdf2image
python-multipart


#Additional dependencies:
poppler (for pdf2image): Install via brew install poppler (macOS), sudo apt-get install poppler-utils (Ubuntu), or download for Windows.
spaCy English model: python -m spacy download en_core_web_sm



#Installation

Clone the repository or create a new project directory.
Set up a virtual environment:python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows


Install dependencies:pip install streamlit fastapi uvicorn easyocr spacy pandas pillow numpy opencv-python pdf2image python-multipart
python -m spacy download en_core_web_sm


Install poppler based on your OS (see Prerequisites).

#Usage
Running the Streamlit UI

Ensure the virtual environment is activated.
Run the application:streamlit run tryapp.py


Open your browser at http://localhost:8501.
Upload a lab report file, enable deskewing if needed, and review the extracted tables.

#Running the FastAPI API

In a separate terminal, activate the virtual environment.
Run the API server:uvicorn tryapp:app --reload --port 8000


Use a tool like curl or Postman to send a POST request to http://localhost:8000/upload with a file.

Example API Request
curl -X POST "http://localhost:8000/upload" -F "file=@example.pdf"

File Structure

tryapp.py: Main script containing Streamlit UI, FastAPI app, and extraction logic.

#Configuration

Deskewing: Toggle the "Enable Deskew" checkbox in the UI for tilted scans.
Output Directory: Results are saved in final_reports and corrections directories (created automatically).



OCR Fails: Check console for warnings about image processing failures. Ensure the file is readable and try increasing DPI in preprocess.
Missing Modules: Install any missing packages as indicated by error messages (e.g., pip install python-multipart).
Type Errors: Verify input files are valid; corrupted PDFs may cause unexpected behavior.

#Contact
For issues or suggestions, please open an issue on the repository or contact the maintainer.
