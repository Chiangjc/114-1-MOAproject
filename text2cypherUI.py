import streamlit as st
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
import pandas as pd
import re

# ===== 初始化 LLM & Retriever，只初始化一次 =====
@st.cache_resource
def init_neo4j_and_llm():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    llm = OllamaLLM(model_name="llama3", model_params={"temperature": 0})
    retriever = Text2CypherRetriever(driver=driver, llm=llm)
    rag = GraphRAG(retriever=retriever, llm=llm)
    return driver, llm, retriever, rag

driver, llm, retriever, rag = init_neo4j_and_llm()

# ===== Streamlit UI =====
st.set_page_config(page_title="Graph RAG 問答系統")
st.title("🧠 GraphRAG x Neo4j 問答系統")
st.markdown("請輸入你的問題，例如：`What drugs are related to asthma?`")

query = st.text_input("輸入問題")
submit = st.button("查詢")

if submit and query:
    with st.spinner("正在生成查詢與回答..."):
        try:
            results = retriever.search(query_text=query)

            if results:
                st.subheader("📄 Cypher 查詢與結果")

                for idx, (cypher_query, records) in enumerate(results):
                    st.markdown(f"##### 🔎 Result {idx + 1}")
                    st.code(cypher_query, language="cypher")

                    if records:
                        parsed_records = []
                        for record in records:
                            try:
                                content = record.content  # RetrieverResultItem 的字串表示
                                match = re.search(r"properties=\{(.+?)\}", content)
                                if match:
                                    properties_str = match.group(1)
                                    props = dict(
                                        item.split(": ", 1)
                                        for item in properties_str.split(", ")
                                        if ": " in item
                                    )
                                    clean_props = {
                                        k.strip().strip("'\""): v.strip().strip("'\"")
                                        for k, v in props.items()
                                    }
                                    parsed_records.append(clean_props)
                                else:
                                    st.warning("❗ 無法從 content 中找到 properties 區塊")
                            except Exception as e:
                                st.warning(f"錯誤解析 record：{e}")
                                st.text(f"Raw record: {record}")

                        if parsed_records:
                            df = pd.DataFrame(parsed_records)
                            if not df.empty:
                                st.markdown("#### 📊 分類表格")

                                if 'node_index' in df.columns:
                                    grouped = df.groupby("node_index")
                                    for name, group in grouped:
                                        st.markdown(f"**來源分類：{name}**")
                                        st.dataframe(group.reset_index(drop=True))
                                else:
                                    st.dataframe(df)
                            else:
                                st.info("⚠️ 無可顯示資料。")
                        else:
                            st.info("⚠️ 沒有成功解析的記錄。")
                    else:
                        st.info("⚠️ 查無記錄。")

            else:
                st.info("⚠️ Retriever 未找到任何結果")

            # Step 2: 執行 GraphRAG 生成回應
            st.subheader("💬 RAG 回答")
            answer = rag.search(query_text=query)
            st.success(answer.answer)

        except Exception as e:
            st.error(f"❌ 發生錯誤：{e}")
