import streamlit as st
import json
import yaml
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os

load_dotenv()

st.title('The Generator')

st.info("일반적인 챗봇입니다. PDF를 업로드하면 PDF 내용을 기반으로 답변합니다. '요약해줘:'로 시작하면 Chain of Density 요약을 수행합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_qa" not in st.session_state:
    st.session_state.pdf_qa = None

def add_message(role, message):
    st.session_state.messages.append(ChatMessage(role=role, content=message))

def print_message():
    for message in st.session_state.messages:
        st.chat_message(message.role).write(message.content)

def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_template = yaml.safe_load(file)
    return prompt_template

def chain_of_density_summary(content, max_words=75, entity_range="3-5", iterations=3):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.1)
    
    prompt_template = load_prompt_template('chain_of_density_prompt.yaml')
    
    system_prompt = prompt_template['system_prompt'].format(
        max_words=max_words,
        entity_range=entity_range,
        iterations=iterations
    )
    
    user_prompt = prompt_template['user_prompt'].format(content=content)

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    summaries = json.loads(response.content)
    return summaries

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 2})
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
            retriever=retriever
        )
        return qa
    finally:
        os.unlink(tmp_file_path)

# 사이드바에 PDF 업로드 기능 추가
with st.sidebar:
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        st.session_state.pdf_qa = process_pdf(uploaded_file)
        st.success("PDF가 성공적으로 업로드되었습니다!")

# 실제로 이전 대화기록을 모두 출력
print_message()

# 채팅 입력창
user_input = st.chat_input("질문하세요")

# 만약에 유저가 채팅을 입력한다면 채팅이 뜨게 만든다
if user_input:
    st.chat_message('user').write(user_input)
    
    if user_input.startswith("요약해줘:"):
        text_to_summarize = user_input[6:].strip()
        summaries = chain_of_density_summary(text_to_summarize)
        
        with st.chat_message('ai'):
            for i, summary in enumerate(summaries):
                st.markdown(f"### 요약 {i+1}")
                st.write(f"Missing Entities: {summary['missing_entities']}")
                st.write(f"Summary: {summary['denser_summary']}")
        
        add_message("user", user_input)
        add_message("ai", json.dumps(summaries, ensure_ascii=False))
    elif st.session_state.pdf_qa:
        result = st.session_state.pdf_qa({"question": user_input, "chat_history": []})
        ai_answer = result['answer']
        
        with st.chat_message('ai'):
            st.write(ai_answer)
        
        add_message("user", user_input)
        add_message("ai", ai_answer)
    else:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.1)
        response = llm.invoke(user_input)
        ai_answer = response.content

        with st.chat_message('ai'):
            st.write(ai_answer)

        add_message("user", user_input)
        add_message("ai", ai_answer)