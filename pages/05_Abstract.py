import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

openai_api_key = os.getenv("OPENAI_API_KEY")

def main():
  # streamlit으로 제목과 input box 생성
  st.title("논문 초록 작성기")
  st.info("논문의 내용을 넣으면 초록을 작성해 줘요")
  content = st.text_input("논문의 초록을 작성할 주제와 간단한 내용을 입력해주세요.")

  # 언어모델 불러오기
  llm = ChatOpenAI(openai_api_key=openai_api_key)
  prompt = ChatPromptTemplate.from_messages(
      [("system", "You are a world class paper writer."), ("user", "{input}")]
  )
  output_parser = StrOutputParser()
  chain = prompt | llm | output_parser

  # 버튼 클릭시 논문 초록 생성
  if st.button("논문 초록 작성하기"):
      with st.spinner("초록 작성 중입니다..."):
          result = chain.invoke({"input": f"{content}에 대한 논문의 초록을 작성해줘."})
          st.write(result)

if __name__ == "__main__":
    main()