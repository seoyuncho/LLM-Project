import gzip
import pickle
import streamlit as st
from openai import OpenAI
from collections import defaultdict

# Show title and description.
st.header("낚시성 기사 제목 분석기")
st.write("신문사별 낚시성 기사 제목 비율을 분석합니다.")

# Load the news data
@st.cache_resource
def load_news_data():
    file_name = "./news_w_embeddings_0731.pickle.gz"
    with gzip.open(file_name, 'rb') as f:
        news_data = pickle.load(f)
    return news_data

# Function to analyze clickbait title
def analyze_clickbait(client, title):
    prompt = f"""Analyze the following news title and determine if it's clickbait:

Title: "{title}"

신문사 기사 제목을 LLM을 활용해서 과장되고 선정적인 낚시성 기사인지 판별할 것입니다. 제목을 분석하고 짧은 설명과 함께 "낚시성 기사" 또는 "낚시성 기사 아님"으로 결론을 내려주세요."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing news titles for clickbait."},
            {"role": "user", "content": prompt}
        ]
    )
    analysis = response.choices[0].message.content
    is_not_clickbait = "낚시성 기사 아님" in analysis.upper()
    return not is_not_clickbait, analysis

# Main function
def main():
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        return

    client = OpenAI(api_key=openai_api_key)
    news_data = load_news_data()

    st.subheader("신문사별 낚시성 기사 분석")

    # Analyze a sample of articles for each news source
    sample_size = st.slider("분석할 기사 수 (신문사당)", min_value=3, max_value=50, value=10)
    
    if st.button("분석 시작"):
        results = defaultdict(lambda: {"total": 0, "clickbait": 0})

        with st.spinner("기사를 분석 중입니다..."):
            for news_group in news_data:
                source = news_group[0].get("provider")

                if results[source]["total"] >= sample_size:
                    continue

                title = news_group[0].get("title")
                   
                is_clickbait, analysis = analyze_clickbait(client, title)
                
                results[source]["total"] += 1
                if is_clickbait:
                    results[source]["clickbait"] += 1

                st.write(f"신문사: {source}")
                st.write(f"제목: {title}")
                st.write(f"분석: {analysis}")
                st.write("---")

        st.subheader("분석 결과")
        for source, data in results.items():
            clickbait_ratio = (data["clickbait"] / data["total"]) * 100 if data["total"] > 0 else 0
            st.write(f"{source}: 낚시성 기사 비율 - {clickbait_ratio:.2f}% ({data['clickbait']}/{data['total']})")

if __name__ == "__main__":
    main()