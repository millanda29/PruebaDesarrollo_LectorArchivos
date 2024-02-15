# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo
import os
# Importa el módulo feedparser para procesar archivos RSS
import feedparser
# Importa la biblioteca Streamlit para crear aplicaciones web interactivas
import streamlit as st
# Importa el CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter
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

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()


# Función para procesar el texto extraído de un archivo HTML o RSS
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


# Función principal de la aplicación
def main():
    st.title("Preguntas a archivos HTML o RSS")

    uploaded_files = st.file_uploader("Sube tus archivos HTML o RSS", type=["html", "rss"], accept_multiple_files=True)

    if uploaded_files:
        text = ""

        for uploaded_file in uploaded_files:
            if uploaded_file.type == "text/html":
                text += uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/rss+xml":
                rss_content = feedparser.parse(uploaded_file)
                for entry in rss_content.entries:
                    text += entry.title + " " + entry.description + " "

        knowledge_base = process_text(text)

        query = st.text_input('Escribe tu pregunta para HTML o RSS...')

        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()

        if query:
            docs = knowledge_base.similarity_search(query)

            model = "gpt-3.5-turbo-instruct"
            temperature = 0

            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": docs})
                st.write(response["output_text"])
                st.text(f'{cost}')


# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
    main()
