import streamlit as st
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
import pandas as pd
import re
import ast

# ===== Streamlit UI =====
st.set_page_config(page_title="Graph RAG 問答系統")
st.title("🧠 GraphRAG x Neo4j 問答系統")
st.markdown("請輸入你的問題，例如：`What drugs are related to asthma?`")

query = st.text_input("輸入問題")

# 模型選擇（分成 Cypher 大模型 + Answer 小模型）
cypher_model = st.selectbox(
    "選擇 Cypher 生成模型",
    ["llama3", "mistral", "gemma:7b", "qwen2:7b"]
)
answer_model = st.selectbox(
    "選擇 回答生成模型",
    ["llama3", "mistral", "tinyllama", "phi3", "gemma:2b"]
)

submit = st.button("查詢")


# ===== 初始化 Neo4j & LLM =====
@st.cache_resource
def init_neo4j_and_llm(cypher_model_name, answer_model_name):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    # 分開 Cypher 與 Answer 模型
    cypher_llm = OllamaLLM(model_name=cypher_model_name, model_params={"temperature": 0})
    answer_llm = OllamaLLM(model_name=answer_model_name, model_params={"temperature": 0})

    retriever = Text2CypherRetriever(driver=driver, llm=cypher_llm)
    rag = GraphRAG(retriever=retriever, llm=answer_llm)
    return driver, cypher_llm, answer_llm, retriever, rag


driver, cypher_llm, answer_llm, retriever, rag = init_neo4j_and_llm(cypher_model, answer_model)

if submit and query:
    with st.spinner("正在生成查詢與回答..."):
        try:
            # Step 1: Execute the retriever search
            results = retriever.search(query_text=query)

            if results:
                st.subheader("📄 Cypher 查詢與結果")

                for idx, (cypher_query, result_data) in enumerate(results):
                    st.markdown(f"##### 🔎 Result {idx + 1}")
                    st.code(cypher_query, language="cypher")

                    # Check if the result_data is a list of records (the expected case)
                    if isinstance(result_data, list):
                        records = result_data
                        if records:
                            parsed_records = []
                            for record in records:
                                if not hasattr(record, 'content') or record.content is None:
                                    continue

                                content = record.content

                                # Strategy 1: Use regex to find and parse properties
                                match = re.search(r"properties=\{(.+?)\}", content)
                                if match:
                                    properties_str = match.group(1)
                                    try:
                                        # Safely parse the properties string into a dictionary
                                        props = ast.literal_eval(f"{{{properties_str}}}")
                                        parsed_records.append(props)
                                    except (ValueError, SyntaxError) as e:
                                        st.warning(f"❗ Failed to parse properties string with ast.literal_eval: {e}")
                                        st.text(f"Raw properties string: {properties_str}")
                                else:
                                    # Strategy 2: Fallback to try parsing the entire content as a dict
                                    try:
                                        content_dict = ast.literal_eval(content)
                                        if isinstance(content_dict, dict):
                                            parsed_records.append(content_dict)
                                        else:
                                            # If it's a simple string, treat it as a result
                                            parsed_records.append({'result': content_dict})
                                    except (ValueError, SyntaxError):
                                        # Strategy 3: If all else fails, treat it as a simple string
                                        parsed_records.append({'result': content})

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
                            
                    # Handle the case where result_data is a metadata dictionary
                    elif isinstance(result_data, dict):
                        st.markdown("#### ⚙️ Metadata")
                        st.json(result_data)
                    else:
                        st.warning(f"❗ Unexpected result format: {type(result_data)}")

            else:
                st.info("⚠️ Retriever 未找到任何結果")

            # Step 2: 執行 GraphRAG 生成回應
            st.subheader("💬 RAG 回答")
            answer = rag.search(query_text=query)
            st.success(answer.answer)

        except Exception as e:
            st.error(f"❌ 發生錯誤：{e}")