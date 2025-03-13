from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os
import PyPDF2
import docx
import io
import re

# Load environment variables
load_dotenv()

# Azure AI Foundry Configuration
endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
model_name = os.getenv("DEPLOYMENT_NAME")
key = os.getenv("AZURE_INFERENCE_SDK_KEY")

if not endpoint or not key:
    raise ValueError("Missing Azure AI Inference credentials in .env file")

client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

app = FastAPI()

# ✅ CORS: Solo permitir solicitudes desde el frontend en Azure
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deepseekmodel-add0agb2abanc9eb.canadacentral-01.azurewebsites.net/"],  # Reemplazar con tu URL real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de procesamiento de archivos y chat"}

# ✅ Función para extraer texto de diferentes formatos de archivo
def extract_text_from_file(file: UploadFile) -> str:
    try:
        content = file.file.read()

        if file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            return " ".join([page.extract_text() or "" for page in pdf_reader.pages])

        elif file.filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return " ".join([para.text for para in doc.paragraphs])

        elif file.filename.endswith(".txt"):
            return content.decode("utf-8")

        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al extraer texto: {str(e)}")

# ✅ Endpoint corregido (ahora coincide con el frontend)
@app.post("/process-file/")
async def process_file(file: UploadFile = File(...)):
    try:
        file_text = extract_text_from_file(file)

        system_message = SystemMessage(content="You are a helpful assistant. Summarize the text provided.")
        user_message = UserMessage(content=f"Resumen del texto: {file_text}")

        response = client.complete(
            messages=[system_message, user_message],
            model=model_name,
            max_tokens=500
        )

        raw_summary = response.choices[0].message.content

        # ✅ Eliminar etiquetas <think>...</think>
        cleaned_summary = re.sub(r"<think>.*?</think>", "", raw_summary, flags=re.DOTALL).strip()

        if not cleaned_summary:
            raise HTTPException(status_code=500, detail="No se pudo extraer un resumen válido.")

        return {"summary": cleaned_summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

# ✅ Chat Endpoint corregido
@app.post("/chat")
async def chat_with_ai(user_input: dict):
    try:
        user_message = user_input.get("message", "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

        system_message = SystemMessage(content="You are a helpful AI assistant.")
        user_chat_message = UserMessage(content=user_message)

        response = client.complete(
            messages=[system_message, user_chat_message],
            model=model_name,
            max_tokens=500
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el chat: {str(e)}")
