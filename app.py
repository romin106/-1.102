import streamlit as st
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import quote
import json
from google import genai
from google.genai import types

st.set_page_config(page_title="기업 리스크 & 경제 뉴스 AI 검색엔진", layout="wide")
st.title("📊 기업 매출 & 악재 리스크 AI 실시간 검색엔진")

# Streamlit Secrets에서 API 키 받아오기
if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("🔑 Streamlit Cloud Settings > Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.stop()

search_query = st.text_input("검색하고 싶은 기업명 또는 경제 키워드를 입력하세요:", "카카오")

if search_query:
    st.subheader(f"🔍 '{search_query}' 실시간 AI 분석 결과")
    
    with st.spinner("최신 경제 뉴스를 수집하고 AI 문맥 분석 진행 중..."):
        news_results = []
        try:
            encoded_query = quote(f"{search_query}")
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
            
            req = urllib.request.Request(
                rss_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            
            response = urllib.request.urlopen(req)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            # 무료 한도(RPM) 고려 및 빠른 속도를 위해 15건 수집
            for item in root.findall('.//item')[:15]:
                title = item.find('title').text if item.find('title') is not None else ''
                link = item.find('link').text if item.find('link') is not None else '#'
                pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ''
                
                news_results.append({
                    'title': title,
                    'snippet': f"발행일: {pubDate}",
                    'link': link
                })
        except Exception as e:
            st.error(f"뉴스 수집 오류: {e}")

    if news_results:
        # 단일 API 요청으로 묶어서 처리 (속도 향상 및 무료 한도 절약)
        titles_list = [f"{idx}. {n['title']}" for idx, n in enumerate(news_results)]
        formatted_titles = "\n".join(titles_list)
        
        prompt = f"""
        당신은 금융 리스크 분석가입니다. 아래 뉴스 제목 목록을 읽고, 해당 뉴스가 언급된 기업 입장에서 '호재', '악재', '중립' 중 무엇에 해당하는지 판단하세요.
        단순 단어 포함 여부가 아닌, 문맥상의 의미를 정확히 파악하여 분류해야 합니다.
        
        [뉴스 목록]
        {formatted_titles}
        
        [응답 형식]
        반드시 부연 설명 없이 아래와 같은 JSON 배열 형태로만 응답하세요:
        [
          {{"id": 0, "sentiment": "호재"}},
          {{"id": 1, "sentiment": "악재"}},
          {{"id": 2, "sentiment": "중립"}}
        ]
        """
        
        sales_news = []
        risk_news = []
        neutral_news = []

        try:
            # 무료 전용 모델 gemini-2.5-flash 사용
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            ai_evaluations = json.loads(response.text)
            
            # AI 판단 결과를 바탕으로 분류 진행
            for eval_item in ai_evaluations:
                idx = eval_item.get("id")
                sentiment = eval_item.get("sentiment")
                
                if idx < len(news_results):
                    item = news_results[idx]
                    if sentiment == "호재":
                        sales_news.append(item)
                    elif sentiment == "악재":
                        risk_news.append(item)
                    else:
                        neutral_news.append(item)
                        
        except Exception as e:
            st.error(f"AI 분석 처리 중 오류 발생: {e}")
            neutral_news = news_results

        # 수치 요약 출력 (키워드가 아닌 AI 분석 기준 건수)
        col1, col2, col3 = st.columns(3)
        col1.metric("수집된 전체 뉴스", f"{len(news_results)}건")
        col2.metric("💰 AI 분류 호재/실적 기사", f"{len(sales_news)}건")
        col3.metric("⚠️ AI 분류 악재/리스크 기사", f"{len(risk_news)}건")

        st.markdown("---")

        # 결과 탭 출력
        tab1, tab2, tab3 = st.tabs(["🚨 AI 감지 악재 경보", "💰 AI 감지 호재 뉴스", "📰 전체 뉴스 목록"])

        with tab1:
            if risk_news:
                st.error(f"AI가 문맥상 악재(리스크)로 판단한 기사입니다. (총 {len(risk_news)}건)")
                for n in risk_news:
                    st.write(f"**[{n['title']}]({n['link']})**")
                    st.caption(n['snippet'])
                    st.write("---")
            else:
                st.info("AI가 감지한 악재 관련 뉴스가 없습니다.")

        with tab2:
            if sales_news:
                st.success(f"AI가 문맥상 호재(실적/성장)로 판단한 기사입니다. (총 {len(sales_news)}건)")
                for n in sales_news:
                    st.write(f"**[{n['title']}]({n['link']})**")
                    st.caption(n['snippet'])
                    st.write("---")
            else:
                st.info("AI가 감지한 호재 관련 뉴스가 없습니다.")

        with tab3:
            for n in news_results:
                st.write(f"**[{n['title']}]({n['link']})**")
                st.caption(n['snippet'])
                st.write("---")
