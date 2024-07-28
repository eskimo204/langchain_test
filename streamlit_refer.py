import streamlit as st
import os
import base64
import uuid
import pytesseract
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

# Streamlit 페이지 설정
st.set_page_config(page_title="PDF/Image Extractor", page_icon="📄")

st.title("PDF 및 이미지 추출기")
st.sidebar.header("파일 업로드 및 설정")

# 파일 업로드
uploaded_file = st.sidebar.file_uploader("파일을 업로드하세요", type=["pdf", "docx", "jpg", "png"])

# OpenAI API 키 입력
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = api_key

if uploaded_file is not None:
    file_path = f"/tmp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if uploaded_file.type == "application/pdf":
        # PDF 파일 처리
        raw_pdf_elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path="/tmp"
        )

        texts, tables = categorize_elements(raw_pdf_elements)
        joined_texts = " ".join(texts)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=0)
        texts_4k_token = text_splitter.split_text(joined_texts)

        # 텍스트 요약
        text_summaries, table_summaries = generate_text_summaries(texts_4k_token, tables, summarize_texts=True)

        # 이미지 처리
        img_base64_list, image_summaries = generate_img_summaries("/tmp")

        # 벡터 스토어 및 검색기 생성
        vectorstore = Chroma(collection_name="sample-rag-multi-modal", embedding_function=OpenAIEmbeddings())
        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            text_summaries,
            texts,
            table_summaries,
            tables,
            image_summaries,
            img_base64_list,
        )

        st.write("PDF 파일에서 텍스트, 테이블 및 이미지 추출 완료.")
    
    elif uploaded_file.type.startswith("image/"):
        # 이미지 파일 처리
        img_base64 = encode_image(file_path)
        prompt = "You are an assistant tasked with summarizing images for retrieval. Give a concise summary of the image that is well optimized for retrieval."
        image_summary = image_summarize(img_base64, prompt)
        
        st.write(f"이미지 요약: {image_summary}")

    st.write("파일 처리가 완료되었습니다.")

# 필요한 함수들
def categorize_elements(raw_pdf_elements):
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

def generate_text_summaries(texts, tables, summarize_texts=False):
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. These summaries will be embedded and used to retrieve the raw text or table elements. Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    text_summaries = []
    table_summaries = []
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    return text_summaries, table_summaries

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4", max_tokens=2048)
    msg = chat.invoke(
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ]
    )
    return msg.content

def generate_img_summaries(path):
    img_base64_list = []
    image_summaries = []
    prompt = """You are an assistant tasked with summarizing images for retrieval. These summaries will be embedded and used to retrieve the raw image. Give a concise summary of the image that is well optimized for retrieval."""
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
    return img_base64_list, image_summaries

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
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
