import streamlit as st
from langchain_core.messages.chat import ChatMessage
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

load_dotenv()
st.title("PDF 기반 문서 챗봇")
st.info("PDF를 넣고 대화해 보시죠")
# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# 메시지를 저장할 list를 생성한다
if "messages" not in st.session_state:
    st.session_state.messages = []

# RAG 체인을 초기화 합니다
# session_state? 뭐고 왜 쓰는거지? -> streamlit은 저장 안 해두면 날라간다 채팅 떄 마다 체인이 돌면 너무 오래걸리지 않겠나
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None 



# 채팅 메시지에 새로운 메시지를 추가하는 함수    
def add_message(role,message) :
    # 메시지 list에 새로운 대화를 추가합니다
    st.session_state.messages.append(ChatMessage(role=role,content=message))
# 이전의 대화기록을 모두 출력하는 함수
def print_message() :
    for message in st.session_state.messages :
        # 대화를 출력
        st.chat_message(message.role).write(message.content)


# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

with st.sidebar :
    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일 업로드", type = ["pdf"])

# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...잠시만 기다려봐")
def embed_file(file) :
    # 업로드한 파일을 캐시 디렉토리에 저장합니다
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    
    embeddings = OpenAIEmbeddings()
    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever

def create_rag_chain(retriever):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("./prompts/pdf-rag.yaml", encoding='utf-8')
    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# 만약에 사용자가 파일을 업로드 했을 때 
if uploaded_file :
    #파일을 캐시에 저장
    retriever = embed_file(uploaded_file)

    # RAG 체인 생성
    st.session_state.rag_chain = create_rag_chain(retriever)

    # 사용자에게 알림
    st.write("파일이 성공적으로 업로드 되었습니다")

# 이전의 대화 내용을 출력
print_message()

# 사용자의 질문 입력
user_input =  st.chat_input("뭐가 궁금해?")

if user_input : 
    # 사용자가 업로드 했을 떄에만 질문에 대한 응답을 하도록 만든다
    if st.session_state["rag_chain"] is not None :
        # 사용자의 질문을 출력
        st.chat_message("user").write(user_input)
        # RAG 체인을 가져옴
        chain = st.session_state["rag_chain"]

        # 체인을 실행시 ai_answer를 받는다
        answer = chain.stream(user_input)

        with st.chat_message('ai') :
            # 빈 공간을 만든다
            chat_container = st.empty()
            # ai의 답변을 출력
            ai_answer = ""
            for token in answer :
                ai_answer += token
                chat_container.markdown(ai_answer)

        # 답변을 출력한다
        # st.chat_message('ai').write(ai_answer)

        # 대화를 추가

        add_message("user", user_input)
        add_message("ai", ai_answer)
    else  :
        st.error("아무것도 안 넣은 것 같은데???")
