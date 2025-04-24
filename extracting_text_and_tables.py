import fitz  # PyMuPDF
import tabula
import json
# import requests
# import os


# def extract_text_from_pdf(pdf_path):
#     # Initialize a list to store text and table data
#     extracted_data = []

#     # Open the PDF file using PyMuPDF
#     pdf_document = fitz.open(pdf_path)

#     # Iterate through each page in the PDF
#     for page_number in range(len(pdf_document)):
#         page = pdf_document[page_number]

#         # Extract text from the page
#         page_text = page.get_text("text")

#         # Check if the page contains tables
#         tables = tabula.read_pdf(pdf_path, pages=page_number + 1, multiple_tables=True, stream=True)

#         # If tables are detected on the page, extract text from tables and format as JSON
#         if tables:
#             table_data = []
#             for table in tables:
#                 table_data.append(table.to_dict(orient="records"))
#             extracted_data.append(page_text.strip())
#             extracted_data.append(table_data)
#         else:
#             # If no tables are detected, store the text as-is
#             extracted_data.append(page_text.strip())

#     # Close the PDF file
#     pdf_document.close()

#     return extracted_data

# # Process PDFs in the folder and get the extracted data
# extracted_data = extract_text_from_pdf(r"C:\Users\CVHS\Downloads\PDF's\JA-020 Salesforce Case Comments and Chatter Functionalities for Novartis Patient Support Center.pdf")

# print(extracted_data)
# # Save the extracted data as JSON
# with open('extracted_data.json', 'w') as json_file:
#     json.dump(extracted_data, json_file, indent=4)

import os
from PyPDF2 import PdfReader
import fitz
import tabula
import json

def extract_text_from_pdf(pdf_path):
    extracted_data = []
    pdf_document = fitz.open(pdf_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        page_text = page.get_text("text")
        tables = tabula.read_pdf(pdf_path, pages=page_number + 1, multiple_tables=True, stream=True)

        if tables:
            table_data = []
            for table in tables:
                table_data.append(table.to_dict(orient="records"))
            extracted_data.append(page_text.strip())
            extracted_data.append(table_data)
        else:
            extracted_data.append(page_text.strip())

    pdf_document.close()
    return extracted_data


extracted_data = extract_text_from_pdf(r"C:\Users\CVHS\Downloads\PDF's\JA-020 Salesforce Case Comments and Chatter Functionalities for Novartis Patient Support Center.pdf")

print(extracted_data)
# Save the extracted data as JSON
with open('extracted_data.json', 'w') as json_file:
    json.dump(extracted_data, json_file, indent=4)

# def process_pdfs_in_folder(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             pdf_path = os.path.join(folder_path, filename)
#             extracted_data = extract_text_from_pdf(pdf_path)

#             # Display extracted data
#             print(f"Extracted Data for {filename}:")
#             print(json.dumps(extracted_data, indent=4))

#             # Save the extracted data as JSON
#             with open(f'{os.path.splitext(filename)[0]}_extracted_data.json', 'w') as json_file:
#                 json.dump(extracted_data, json_file, indent=4)

# if __name__ == "__main__":
#     # Specify the folder path containing the PDFs
#     folder_path = r'C:\Users\DELL\Desktop\GEMINI\upload'

#     # Process PDFs in the folder and get the extracted data
#     process_pdfs_in_folder(folder_path)