# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo
import os
# Importa la clase PdfReader del módulo PyPDF2 para leer archivos PDF
from PyPDF2 import PdfReader
# Importar HTML
from bs4 import BeautifulSoup
import feedparser
# Importa la biblioteca Streamlit para crear aplicaciones web interactivas
import streamlit as st
# Importa el CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter, HTMLHeaderTextSplitter
# Importa OpenAIEmbeddings del módulo langchain.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# Importa FAISS del módulo langchain para realizar búsqueda de similitud
from langchain import FAISS
# Importa load_qa_chain del módulo langchain.chains.question_answering para cargar cadenas de preguntas y respuestas
from langchain.chains.question_answering import load_qa_chain
# Importa OpenAI del módulo langchain.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain.llms import OpenAI
# Importa get_openai_callback del módulo langchain.callbacks para obtener realimentación de OpenAI
from langchain.callbacks import get_openai_callback
# Importa el módulo langchain
import langchain
import feedparser

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de un archivo PDF
def process_text(text):
    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

def process_rss(rss_content):
    # Parsea el contenido RSS
    feed = feedparser.parse(rss_content)

    # Inicializa una cadena para almacenar el texto procesado
    processed_text = ""

    # Itera sobre las entradas del feed RSS
    for entry in feed.entries:
        # Extrae el título y el resumen de cada entrada
        title = entry.title if hasattr(entry, 'title') else ""
        summary = entry.summary if hasattr(entry, 'summary') else ""

        # Concatena el título y el resumen para formar el texto procesado
        processed_text += f"{title}\n{summary}\n\n"

    return processed_text

def process_html(html_string):
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ])
    html_header_splits = html_splitter.split_text(html_string)
    return html_header_splits

def process_files(files):
    combined_text = ""

    for file in files:
        if file.type == "text/html":
            html_content = file.read()
            soup = BeautifulSoup(html_content, "html.parser")
            combined_text += soup.get_text(separator="\n")
        elif file.type == "application/rss+xml":
            rss_content = file.read().decode("utf-8")  # Lee y decodifica el contenido del archivo RSS
            combined_text += process_rss(rss_content)
        else:
            st.warning(f"El formato de archivo {file.type} no es compatible.")

    knowledge_base = process_text(combined_text)
    return knowledge_base

def main():
    st.title("Preguntas a archivos HTML")

    html_files = st.file_uploader("Sube tus archivos HTML", type=["html", "rss"], accept_multiple_files=True)

    if html_files is not None and html_files:
        # Combina el texto de todos los archivos HTML subidos
        knowledge_base = process_files(html_files)

    query = st.text_input('Escribe tu pregunta para el PDF...')

    cancel_button = st.button('Cancelar')

    if cancel_button:
        st.stop()

    if query:
        # Realiza una búsqueda de similitud en la base de conocimientos
        docs = knowledge_base.similarity_search(query)

        # Inicializa un modelo de lenguaje de OpenAI y ajusta sus parámetros
        model = "gpt-3.5-turbo-instruct"
        temperature = 0

        llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

        # Carga la cadena de preguntas y respuestas
        chain = load_qa_chain(llm, chain_type="stuff")

        # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
        with get_openai_callback() as cost:
            response = chain.invoke(input={"question": query, "input_documents": docs})
            print(cost)
            st.write(response["output_text"], unsafe_allow_html=True)
            st.text(f'{cost}')

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
    main()
