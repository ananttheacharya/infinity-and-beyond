import PyPDF2
import os

pdf_files = [
    "Physics-Informed_Machine_Learning_for_Intelligent_.pdf",
    "energies-18-05523-v2.pdf"
]

for pdf_file in pdf_files:
    print(f"\n--- Extracting Abstract/Intro from: {pdf_file} ---")
    try:
        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            # Read first 2 pages
            text = ""
            for i in range(min(2, len(reader.pages))):
                text += reader.pages[i].extract_text()
            
            # Print a snippet of the text (first 2000 characters)
            print(text[:2000])
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
