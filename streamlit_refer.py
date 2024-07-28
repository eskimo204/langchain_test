import os
import streamlit as st
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
# from langchain.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import base64
import pytesseract
import io
import re
import uuid

st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📄")

def extract_pdf_elements(file, extract_images_in_pdf=True):
    import fitz  # PyMuPDF
    from pdfminer.high_level import extract_text

    doc = fitz.open(stream=file.read(), filetype="pdf")
    extracted_text = extract_text(io.BytesIO(file.getvalue()))
    extracted_images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_path = f"page_{page_num + 1}_image_{img_index + 1}.png"
            image.save(image_path)
            extracted_images.append(image_path)

    return extracted_text, extracted_images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4", max_tokens=2048)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ])
    ])
    return msg.content

def generate_img_summaries(path):
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""
    img_base64_list = []
    image_summaries = []

    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries

def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever

st.title("Chat with your PDFs")
st.sidebar.header("Upload your files")
uploaded_files = st.sidebar.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    if uploaded_files:
        all_texts = []
        all_images = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".pdf"):
                extracted_text, extracted_images = extract_pdf_elements(uploaded_file)
                all_texts.append(extracted_text)
                all_images.extend(extracted_images)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts_4k_token = text_splitter.split_text(" ".join(all_texts))

        vectorstore = FAISS.from_texts(texts_4k_token, OpenAIEmbeddings())
        retriever = create_multi_vector_retriever(
            vectorstore,
            texts_4k_token,
            all_texts,
            [],
            [],
            [],
            all_images
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain(
            retriever=retriever,
            memory=memory,
            llm=OpenAI(model="gpt-3.5-turbo")
        )

        st.text_input("Ask something", key="user_input", on_change=lambda: qa_chain({"question": st.session_state.user_input}))
        
        for message in st.session_state.chat_history:
            st.write(f"**{message['author']}**: {message['content']}")
else:
    st.warning("Please enter your OpenAI API key.")
