import streamlit as st
from langchain_teddynote.models import MultiModal
from langchain_openai import ChatOpenAI
from langchain_core.messages.chat import ChatMessage
st.title("이미지 인식 gpt")
import os
from dotenv import load_dotenv
load_dotenv()

st.info("이미지 인식은 돈이 많이 들어요 비상시에만 쓰시죠")
# 메시지를 저장할 list를 생성한다
if "messages" not in st.session_state:
    st.session_state.messages = []


# 채팅 메시지에 새로운 메시지를 추가하는 함수    
def add_message(role,message) :
    # 메시지 list에 새로운 대화를 추가합니다
    st.session_state.messages.append(ChatMessage(role=role,content=message))
# 이전의 대화기록을 모두 출력하는 함수
def print_message() :
    for message in st.session_state.messages :
        # 대화를 출력
        st.chat_message(message.role).write(message.content)

with st.sidebar :
    url = st.text_input("이미지 url을 입력해주세요")
    system_prompt = st.text_area("시스템 프롬프트",""""당신은 이미지 를 해석하는 금융 AI 어시스턴트 입니다. 
당신의 임무는 주어진  이미지 바탕으로  사실을 정리하여 친절하게 답변하는 것입니다.""")
if url : 
    st.image(url)

def create_chain(system_prompt, user_prompt) :
    # 객체 생성
    llm = ChatOpenAI(temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",openai_api_key=os.getenv("OPENAI_API_KEY") # 모델명
)
    #멀티모달 객체 생성
    chain = MultiModal(llm, system_prompt=system_prompt, user_prompt= user_prompt)
    return chain
    

# 이전 대화 기록을 모두 출력
print_message()
# 채팅 입력창
user_input  = st.chat_input("질문하세요")

# 만약에 유저가 채팅을 입력한다면 채팅이 뜨게 만든다
if user_input : 
    st.chat_message('user').write(user_input)
    #체인 생성
    chain = create_chain(system_prompt,user_input)
    # 체인을 실행시 ai_answer를 받는다
    answer = chain.stream(url)

    with st.chat_message('ai') :
        # 빈 공간을 만든다
        chat_container = st.empty()
        # ai의 답변을 출력
        ai_answer = ""
        for token in answer :
            ai_answer += token.content
            chat_container.markdown(ai_answer)

    # 답변을 출력한다
    # st.chat_message('ai').write(ai_answer)

    # 대화를 추가

    add_message("user", user_input)
    add_message("ai", ai_answer)
