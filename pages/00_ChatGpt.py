import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
# streamlit을 통해  웹사이트를 만들어준다

st.title('The Generator')

st.info("일반적인 챗봇입니다")


# streamlit의 단점 -> 모든 페이지가 전부 코드가 새로고침되어 실행된다 -> 채팅이 쌓이지가 않는다
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
    # 사용자가 선택한 프롬프트
    import glob
    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox(
        "프롬프트 선택",prompt_files,
        index=0)
    # 모델 선택
    select_model = st.selectbox("모델 선택", ["gpt-4o-mini","gpt-4o"],index=0)

# 체인 생성
def create_chain(selected_prompt,select_model) :
    # 사용자의 프롬프트를 정의
    # prompt = PromptTemplate.from_template("""당신은 친절하고 상냥한 ai 챗봇입니다 사용자의 질문에 답변하세요.
    #                                     질문 : {question}   
    #                                       """)
    prompt = load_prompt(selected_prompt, encoding='utf-8')
    # LLM 정의
    llm  = ChatOpenAI(model_name = select_model, temperature=0.1)
    # 체인 생성
    chain = prompt | llm | StrOutputParser()
    return chain
                                        
# 실제로 이전 대화기록을 모두 출력
print_message()

# 채팅 입력창
user_input  = st.chat_input("질문하세요")

# 만약에 유저가 채팅을 입력한다면 채팅이 뜨게 만든다
if user_input : 
    st.chat_message('user').write(user_input)
    #체인 생성
    chain = create_chain(selected_prompt,select_model)
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


