import sys
import os
#import time

from google.cloud import logging
from google.cloud import storage
import gradio as gr
import vertexai.generative_models

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

#ProjectInfo
PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
GCP_BUCKET=os.environ.get("GCP_BUCKET") #The Bucket were the uploaded files will be stored

# PROJECT_ID = "arctic-analyzer-435209-m1"  # @param {type:"string"}
# LOCATION = "europe-west3"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)

#Logging
client = logging.Client(project=PROJECT_ID)
client.setup_logging()
log_name = "genai-vertex-text-log"
logger = client.logger(log_name)

#MODEL
MODEL_ID = "gemini-1.5-pro"  # @param {type:"string"}
#MODEL_ID = "gemini-1.5-flash"  # @param {type:"string"}
# Load a example model without system instructions
##model = GenerativeModel(MODEL_ID)
# Load a example model with system instructions
model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a helpful assistant.",
    ],
)


def generate_results(prompt,pdf_file1,pdf_file2):
    logger.log_text(prompt)
    pdf_file1_uri= upload_pdf_to_gcs(pdf_file1,"mypdf1.pdf")
    pdf_file2_uri= upload_pdf_to_gcs(pdf_file2,"mypdf2.pdf")
    #print(pdf_file1_uri)
    #pdf_file_uri="gs://demo_gradio_data/test.pdf"
    pdf_file1_part = Part.from_uri(pdf_file1_uri, mime_type="application/pdf")
    pdf_file2_part = Part.from_uri(pdf_file2_uri, mime_type="application/pdf")
    contents = [prompt, pdf_file1_part,pdf_file2_part]
    response = model.generate_content(contents)
    return response.text 


def upload_pdf_to_gcs(pdf_file,file_name):
    """Uploads a PDF file to a GCS bucket.
    Args:
       pdf_file: The uploaded PDF file.
       # bucket_name: The name of the GCS bucket.
       # file_name: The desired name for the file in the bucket.
    """
    bucket_name=GCP_BUCKET
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
   # Open the PDF file and read its contents
    with open(pdf_file, 'rb') as f:  # Open in binary mode ('rb') for PDF files
        pdf_data = f.read()
    pdf_file_uri=f"gs://{bucket_name}/{file_name}"
    # Upload the file to GCS
    blob.upload_from_string(pdf_data, content_type='application/pdf')
    # Add a 1-second delay
    #time.sleep(3)
    return pdf_file_uri

demo = gr.Interface(
    fn=generate_results,
    inputs=[
        gr.Textbox(
            label="Enter prompt:",
            value="What are the main differences in the two documents?",
        ),
        gr.File(label="Upload PDF File1"),
        gr.File(label="Upload PDF File2"),
        # gr.Textbox(label="PDF File Name"),
        # gr.Slider(32, 1024, value=512, step=32, label="max_output_tokens"),
        # gr.Slider(0, 1, value=0.2, step=0.1, label="temperature"),
        # gr.Slider(0, 1, value=0.8, step=0.1, label="top_p"),
        # gr.Slider(1, 40, value=38, step=1, label="top_k"),
    ],
    outputs="text",
    #examples=examples,
    #theme=gr.themes.Soft(),
    theme=gr.themes.Default(primary_hue="orange"),
)
demo.launch(server_name="0.0.0.0", server_port=8080,debug=True)

     