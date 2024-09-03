import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

st.title("환영합니다!")

# API Key 입력란
open_api_key = st.text_input("OpenAI API Key 입력", type="password")
tavily_api_key = st.text_input("Tavily API Key 입력", type="password")

# OpenAI API Key 설정 버튼
if st.button("OpenAI API Key 설정"):
    if open_api_key:
        # 환경 변수 설정
        os.environ["OPENAI_API_KEY"] = open_api_key
        st.success("OpenAI API 키가 설정되었습니다")
    else:
        st.error("유효한 OpenAI API 키를 입력해주세요")

# Tavily API Key 설정 버튼
if st.button("Tavily API Key 설정"):
    if tavily_api_key:
        # 환경 변수 설정
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        st.success("Tavily API 키가 설정되었습니다")
    else:
        st.error("유효한 Tavily API 키를 입력해주세요")
