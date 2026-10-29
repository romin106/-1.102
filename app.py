import streamlit as st
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import quote
import json
import re
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.6-flash"  # 사용 가능한 모델명으로 필요 시 변경
NEWS_COUNT = 15

st.set_page_config(page_title="기업 주가 전망 AI", layout="wide")
st.title("📈 뉴스 + 실제 주가 기반 주가 전망 AI")
st.caption("최신 뉴스와 실제 주가 흐름(이동평균·RSI 등)을 함께 분석해 단기 방향을 추정합니다.")

# ---------- API 키 ----------
if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("🔑 Streamlit Cloud Settings > Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.stop()


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


# ---------- 뉴스 수집 ----------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(query: str, limit: int = NEWS_COUNT):
    encoded = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        root = ET.fromstring(res.read())

    results = []
    for item in root.findall(".//item")[:limit]:
        results.append({
            "title": item.findtext("title", default=""),
            "link": item.findtext("link", default="#"),
            "snippet": f"발행일: {item.findtext('pubDate', default='')}",
        })
    return results


# ---------- 기업명 -> 티커 ----------
@st.cache_data(ttl=86400, show_spinner=False)
def resolve_ticker(company: str):
    prompt = f"""
'{company}'의 Yahoo Finance 티커를 알려주세요.
- 코스피 상장: 종목코드 6자리 + .KS (예: 삼성전자 005930.KS)
- 코스닥 상장: 종목코드 6자리 + .KQ (예: 에코프로 086520.KQ)
- 미국 등 해외: 해당 티커 (예: 테슬라 TSLA)
- 비상장이거나 기업이 아니거나 확실하지 않으면 ticker를 null로 하세요. 추측하지 마세요.
설명 없이 JSON으로만 응답: {{"ticker": "005930.KS", "name": "삼성전자"}}
"""
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = parse_json(res.text)
        return data.get("ticker"), data.get("name")
    except Exception:
        return None, None


# ---------- 주가 데이터 + 지표 ----------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_price(ticker: str):
    candidates = [ticker]
    # 코스피/코스닥 혼동 대비
    if ticker.endswith(".KS"):
        candidates.append(ticker[:-3] + ".KQ")
    elif ticker.endswith(".KQ"):
        candidates.append(ticker[:-3] + ".KS")

    for t in candidates:
        try:
            df = yf.Ticker(t).history(period="1y", auto_adjust=True)
        except Exception:
            continue
        if df is not None and len(df) >= 30:
            return t, df
    return ticker, None


def compute_indicators(df: pd.DataFrame):
    close = df["Close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    def ret(n):
        return (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100 if len(close) > n else None

    last = float(close.iloc[-1])
    high52, low52 = float(close.max()), float(close.min())
    vol_ratio = float(df["Volume"].iloc[-1] / df["Volume"].tail(20).mean()) if df["Volume"].tail(20).mean() else None

    return {
        "last": last,
        "ret_1d": ret(1),
        "ret_5d": ret(5),
        "ret_20d": ret(20),
        "ret_60d": ret(60),
        "ma20": float(ma20.iloc[-1]),
        "ma60": float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else None,
        "rsi": float(rsi.iloc[-1]),
        "pos_52w": (last - low52) / (high52 - low52) * 100 if high52 > low52 else None,
        "vol_ratio": vol_ratio,
        "chart": pd.DataFrame({"종가": close, "MA20": ma20, "MA60": ma60}).tail(180),
    }


def fmt(v, suffix="", nd=1):
    return "N/A" if v is None or pd.isna(v) else f"{v:.{nd}f}{suffix}"


def price_summary_text(ind: dict) -> str:
    trend = "정배열(단기>중기)" if ind["ma60"] and ind["ma20"] > ind["ma60"] else "역배열(단기<중기)"
    return (
        f"현재가 {fmt(ind['last'], nd=2)}, 전일 대비 {fmt(ind['ret_1d'], '%')}, "
        f"5일 {fmt(ind['ret_5d'], '%')}, 20일 {fmt(ind['ret_20d'], '%')}, 60일 {fmt(ind['ret_60d'], '%')}, "
        f"MA20 {fmt(ind['ma20'], nd=2)}, MA60 {fmt(ind['ma60'], nd=2)} ({trend}), "
        f"RSI(14) {fmt(ind['rsi'])}, 1년 범위 내 위치 {fmt(ind['pos_52w'], '%', 0)}, "
        f"거래량은 20일 평균의 {fmt(ind['vol_ratio'], '배', 2)}"
    )


# ---------- AI 종합 분석 ----------
@st.cache_data(ttl=600, show_spinner=False)
def analyze(company: str, titles: tuple, price_text: str):
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
    price_block = price_text if price_text else "주가 데이터 없음 (뉴스만으로 판단)"
    prompt = f"""
당신은 주식 시장 애널리스트입니다. '{company}'에 대한 최신 뉴스와 실제 주가 지표가 아래에 있습니다.

[작업]
1. 각 뉴스를 주가 영향 기준으로 '호재', '악재', '중립'으로 분류하고 영향도(impact)를 1~3으로 매기세요.
2. 뉴스(재료)와 주가 지표(추세·과열 여부)를 함께 고려해 단기(1~2주) 방향을
   '상승', '하락', '보합' 중 하나로 전망하세요.
   - 호재가 많아도 RSI 70 이상 과열이거나 이미 급등했다면 선반영 가능성을 감안하세요.
   - 악재가 많아도 RSI 30 이하 과매도라면 반등 가능성도 감안하세요.
   - 뉴스와 지표가 엇갈리면 확신도를 낮추세요.
3. 확신도(confidence)는 0~100 정수. 기업/경제 키워드가 아니라 판단이 어려우면 outlook을 '판단불가'로 하세요.

[뉴스 목록]
{numbered}

[주가 지표]
{price_block}

[응답 형식] 설명 없이 JSON 객체로만 응답:
{{
  "items": [{{"id": 0, "sentiment": "호재", "impact": 2}}],
  "outlook": "상승",
  "confidence": 65,
  "summary": "뉴스와 주가 흐름을 함께 고려한 근거 2~3문장",
  "price_comment": "주가 지표만 본 한 줄 해석",
  "up_factors": ["상승 요인1"],
  "down_factors": ["하락 요인1"]
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return parse_json(response.text)


# ---------- UI ----------
col_a, col_b = st.columns([3, 1])
search_query = col_a.text_input("기업명을 입력하세요 (예: 삼성전자, 카카오, 테슬라):", "카카오")
manual_ticker = col_b.text_input("티커 직접 입력 (선택)", "", help="예: 035720.KS, AAPL")

if search_query:
    with st.spinner("뉴스 수집, 주가 조회, AI 분석 중입니다..."):
        try:
            news_results = fetch_news(search_query)
        except Exception as e:
            st.error(f"뉴스 수집 오류: {e}")
            st.stop()

        if not news_results:
            st.warning("관련 뉴스를 찾지 못했습니다. 다른 검색어를 입력해보세요.")
            st.stop()

        # 티커 확정 -> 주가 조회
        ticker, ticker_name = (manual_ticker.strip(), None) if manual_ticker.strip() else resolve_ticker(search_query)
        ind, used_ticker = None, None
        if ticker:
            used_ticker, price_df = fetch_price(ticker)
            if price_df is not None:
                ind = compute_indicators(price_df)

        try:
            result = analyze(
                search_query,
                tuple(n["title"] for n in news_results),
                price_summary_text(ind) if ind else "",
            )
        except Exception as e:
            st.error(f"AI 분석 처리 중 오류 발생: {e}")
            st.stop()

    # 기사 분류
    sales_news, risk_news, score = [], [], 0
    for ev in result.get("items", []):
        idx = ev.get("id")
        if not isinstance(idx, int) or not (0 <= idx < len(news_results)):
            continue
        w = ev.get("impact") if isinstance(ev.get("impact"), int) else 1
        item = {**news_results[idx], "impact": w}
        if ev.get("sentiment") == "호재":
            sales_news.append(item)
            score += w
        elif ev.get("sentiment") == "악재":
            risk_news.append(item)
            score -= w

    outlook = result.get("outlook", "판단불가")
    confidence = result.get("confidence", 0)

    # ---------- 핵심 결과 ----------
    st.subheader(f"🔍 '{search_query}' 주가 전망")
    icon = {"상승": "📈", "하락": "📉", "보합": "➖"}.get(outlook, "❓")
    box = {"상승": st.success, "하락": st.error}.get(outlook, st.info)
    box(f"## {icon} AI 종합 전망: **{outlook}**  (확신도 {confidence}%)")
    st.write(result.get("summary", ""))

    # ---------- 실제 주가 ----------
    if ind:
        st.markdown(f"#### 💹 실제 주가 현황 ({used_ticker})")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("현재가", fmt(ind["last"], nd=2), f"{fmt(ind['ret_1d'], '%', 2)}")
        p2.metric("5일 수익률", fmt(ind["ret_5d"], "%"))
        p3.metric("20일 수익률", fmt(ind["ret_20d"], "%"))
        p4.metric("RSI(14)", fmt(ind["rsi"]))
        p5.metric("1년 범위 위치", fmt(ind["pos_52w"], "%", 0))
        st.line_chart(ind["chart"])
        if result.get("price_comment"):
            st.info(f"📊 {result['price_comment']}")
    else:
        st.warning(
            "실제 주가 데이터를 가져오지 못해 뉴스만으로 판단했습니다. "
            "비상장사이거나 티커를 찾지 못했을 수 있으니 오른쪽 칸에 티커를 직접 입력해보세요."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🟢 상승 요인**")
        for f in result.get("up_factors", []):
            st.write(f"- {f}")
    with c2:
        st.markdown("**🔴 하락 요인**")
        for f in result.get("down_factors", []):
            st.write(f"- {f}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("수집 뉴스", f"{len(news_results)}건")
    m2.metric("💰 호재", f"{len(sales_news)}건")
    m3.metric("⚠️ 악재", f"{len(risk_news)}건")
    m4.metric("영향도 가중 점수", f"{score:+d}")

    st.warning(
        "⚠️ 이 결과는 뉴스와 과거 주가 지표를 바탕으로 한 AI의 추정이며 투자 권유가 아닙니다. "
        "실제 주가는 실적, 수급, 거시경제 등 다양한 요인에 좌우되므로 투자 판단과 책임은 본인에게 있습니다."
    )
    st.markdown("---")

    # ---------- 탭 ----------
    tab1, tab2, tab3 = st.tabs(["🚨 악재 뉴스", "💰 호재 뉴스", "📰 전체 뉴스"])

    def show(items, empty_msg):
        if not items:
            st.info(empty_msg)
            return
        for n in items:
            st.write(f"**[{n['title']}]({n['link']})**")
            st.caption(n["snippet"])
            st.write("---")

    with tab1:
        show(risk_news, "AI가 감지한 악재 뉴스가 없습니다.")
    with tab2:
        show(sales_news, "AI가 감지한 호재 뉴스가 없습니다.")
    with tab3:
        show(news_results, "뉴스가 없습니다.")
