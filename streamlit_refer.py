import streamlit as st  # Streamlit 라이브러리를 임포트하여 웹 애플리케이션을 쉽게 만듭니다.
import tiktoken  # tiktoken 라이브러리를 임포트하여 텍스트를 토큰화합니다.
from loguru import logger  # loguru 라이브러리에서 logger를 임포트하여 로그 메시지를 기록합니다.

from langchain.chains import ConversationalRetrievalChain  # 대화형 정보 검색 체인을 생성하는 모듈을 임포트합니다.
from langchain.chat_models import ChatOpenAI  # OpenAI의 GPT-3.5-turbo 모델을 사용하여 채팅 응답을 생성하는 클래스입니다.
from langchain_community import chat_models  # LangChain 커뮤니티에서 제공하는 채팅 모델 모듈입니다.
from langchain.chat_models import ChatOpenAI  # 위와 동일한 모듈로 중복된 임포트입니다. 위의 chat_models에서 제거해도 됩니다.

from langchain.document_loaders import PyPDFLoader  # PDF 문서를 로드하고 텍스트를 추출하는 클래스입니다.
from langchain.document_loaders import Docx2txtLoader  # DOCX 문서를 로드하고 텍스트를 추출하는 클래스입니다.
from langchain.document_loaders import UnstructuredPowerPointLoader  # PPTX 문서를 로드하고 텍스트를 추출하는 클래스입니다.

from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트를 재귀적으로 분할하여 작은 청크로 만드는 클래스입니다.
from langchain.embeddings import HuggingFaceEmbeddings  # Hugging Face 모델을 사용하여 텍스트 임베딩을 생성하는 클래스입니다.

from langchain.memory import ConversationBufferMemory  # 대화의 메모리를 관리하는 클래스입니다.
from langchain.vectorstores import FAISS  # Facebook AI에서 제공하는 효율적인 유사도 검색 라이브러리입니다.

from langchain.callbacks import get_openai_callback  # OpenAI API 호출의 콜백을 관리하는 함수입니다.
from langchain.memory import StreamlitChatMessageHistory  # Streamlit에서 채팅 메시지 히스토리를 관리하는 클래스입니다.


def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")
    #페이지 이름 및 아이콘

    st.title("_Private Data :red[QA Chat]_ :books:")
    #페이지의 제목

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    #Streamlit 페이지의 설정과 타이틀을 정의합니다, 추후에 사용될 변수들을 초기화합니다.(사실 정의하는 역할)

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    #사이드바에서 파일 업로드 및 OpenAI API 키를 입력받습니다.

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        #OPEN API kEY를 입력받아야함
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 
        st.session_state.processComplete = True
    #process 버튼을 눌렀을 때, 파일의 텍스트를 추출하고, 텍스트를 청크로 분할한 후, 벡터 스토어를 생성합니다. 
    #그 후, 사용자가 질문을 입력할때 문서에서 관련 정보를 검색하고 답변을 생성하는데 사용되는, 벡터 스토어를 기반으로 한 대화 체인을 생성합니다

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
    #세션 상태에 미시지가 없으면 초기 메시지를 추가합니다. (가장 초기 화면)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #메시지를 화면에 표시합니다. (대화를 주고 받을때, 메시지가 화면에 보이게 함)
    #for 문을 통해 구현하였고, break와 같은 반복문을 종료하는 코드가 없는것으로 보아, 무한히 대화가 생성되도록 함

    history = StreamlitChatMessageHistory(key="chat_messages")
    #히스토리를 가져야 LLM이 이전 대화 내용을 알고, 질문의 문맥을 파악할 수 있음


    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
    #질문 창에 떠 있는 메시지
        with st.chat_message("user"):
            st.markdown(query)
    #사용자(유저)의 질문을 받고
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
    #챗본(assistant)의 답변이 출력된다.
            with st.spinner("Thinking..."): #로딩중...
                result = chain({"question": query})
                #LLM의 답변을 result에 담는다.
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
#사용자가 입력한 질문을 처리하고, GPT 모델을 사용하여 응답을 생성합니다. 생성된 응답과 소스 문서를 표시합니다.

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)
#토큰 개수를 기준으로 텍스트를 나누는 함수

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list
#업로드된 파일의 텍스트를 추출하여 리스트로 반환합니다.

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks
#텍스트를 청크로 분할하여 반환합니다.

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb
#텍스트 청크를 임베딩하고 벡터 스토어를 생성합니다.


def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain
#벡터 스토어와 OpenAI API 키를 사용하여 대화 체인을 생성합니다.


if __name__ == '__main__':
    main()
#main 함수를 실행하여 애플리케이션을 시작
