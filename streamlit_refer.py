import streamlit as st
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import ChatOpenAI
from langchain.chat_models import ChatOpenAI 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
import base64
import io
import uuid
from typing import List, Dict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image as ImageElement
from langchain.callbacks import get_openai_callback
from langchain.memory.chat_message_history import StreamlitChatMessageHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated get_text function
def get_text(docs):
    doc_list = []
    images = []

    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
        documents = loader.load_and_split()
        
        # Extract images
        elements = partition_pdf(file_name, extract_images_in_pdf=True)
        for element in elements:
            if isinstance(element, ImageElement):
                images.append(element)
            else:
                doc_list.append(element)

    return doc_list, images

# Encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Updated get_vectorstore function
def get_vectorstore(text_chunks, image_paths):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)

    img_base64_list = [encode_image(img) for img in image_paths]
    image_embeddings = [embeddings.embed_document(base64.b64decode(img)) for img in img_base64_list]

    store = InMemoryStore()
    vectorstore = Chroma(collection_name="sample-rag-multi-modal", embedding_function=embeddings)
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    
    # Helper function to add documents to vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [Document(page_content=s, metadata={"doc_id": doc_ids[i]}) for i, s in enumerate(doc_summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_chunks:
        add_documents(retriever, [chunk.page_content for chunk in text_chunks], text_chunks)
    
    if img_base64_list:
        add_documents(retriever, img_base64_list, image_embeddings)

    return retriever

# Updated get_conversation_chain function
def get_conversation_chain(retriever, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

# Display images in the chat
def display_images(images):
    for img_base64 in images:
        st.image(base64.b64decode(img_base64))

# Main function updated to handle images
def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")
    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        files_text, images = get_text(uploaded_files)
        text_chunks = [Document(page_content=str(doc)) for doc in files_text]
        retriever = get_vectorstore(text_chunks, [img.uri for img in images])

        st.session_state.conversation = get_conversation_chain(retriever, openai_api_key)
        st.session_state.processComplete = True
        st.session_state.text_chunks = text_chunks

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if st.session_state.text_chunks:
        st.write("### 텍스트 청크 출력")
        for chunk in st.session_state.text_chunks:
            st.text(chunk.page_content)

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(f"**Source:** {doc.metadata['source']}")
                        st.markdown(f"**Content:** {doc.page_content}")

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
