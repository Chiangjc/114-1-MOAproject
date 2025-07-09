import streamlit as st
import requests
from neo4j import GraphDatabase
import re

# 連線設定
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))


# === 🔧 自動修正：移除 Markdown 語法 ===
def clean_cypher_output(raw: str) -> str:
    return raw.strip().removeprefix("```cypher").removesuffix("```").strip()


# === 🔧 自動修正：錯誤語法轉安全版本 ===
def sanitize_cypher(cypher: str) -> str:
    # 檢查是否有 | 與帶引號的關係
    if re.search(r"\[:.*\|.*(\"|`)", cypher):
        match = re.search(r"\((\w+):(\w+)\).*?\((\w+):(\w+)\)", cypher)
        if not match:
            return cypher
        left_var, left_type, right_var, right_type = match.groups()
        rels = re.findall(r'[:|]`?["]?([\w\- ]+)["]?`?', cypher)
        return (
            f"MATCH ({left_var}:{left_type})-[r]->({right_var}:{right_type} {{node_name: 'asthma'}})\n"
            f"WHERE type(r) IN {rels}\n"
            f"RETURN DISTINCT {left_var}.node_name AS result"
        )
    return cypher


# === 🔮 LLM 查詢產生 ===
def generate_cypher_query(user_question, model="llama3"):
    prompt = f"""
你是一個擅長 Neo4j 的查詢專家。以下是知識圖的結構：

【節點】
- anatomy, disease, drug, effect__phenotype, biological_process, molecular_function, cellular_component, exposure, pathway, gene__protein
- 每個節點都有：node_name, node_id, node_source, node_index

【關係】
- (drug)-[:indication]->(disease)(指藥物正式被批准用來治療的疾病或病症)
- (drug)-[:"off-label use"]->(disease)(指藥物未經官方正式核准，但臨床上仍被醫生用來治療的疾病。通常是因為有研究或臨床經驗支持該用途。)
- (drug)-[:contraindication]->(disease)(指不應該使用該藥物的情況，通常是因為可能導致嚴重副作用或危險。)
- (drug)-[:drug_protein]->(gene__protein)
- (gene__protein)-[:protein_protein]->(gene__protein)
- (drug)-[:drug_drug]->(drug)

⚠️ 若關係名稱包含空格，或查詢多種關係類型（如 "indication", "off-label use", "contraindication"），請使用以下寫法：

MATCH (a)-[r]->(b)
WHERE type(r) IN ["indication", "off-label use"] OR ...

don't use '|'

請根據以下問題產出正確 Cypher 查詢：
\"\"\"
{user_question}
\"\"\"
請只輸出 Cypher 查詢，不要加上說明。
"""
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return res.json()["response"].strip()


# === 🧠 LLM 中文自然語言回答 ===
def generate_answer_with_model(question: str, results: list[str], model: str = "mistral") -> str:
    prompt = f"""
請根據下列資訊，用中文說明這些查詢結果與問題的關聯。

使用者問題：
{question}

查詢結果：
{', '.join(results)}

請產出自然語言回覆：
"""
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return res.json()["response"].strip()


# === 📡 執行 Cypher 查詢 ===
def run_cypher_query(cypher):
    with driver.session() as session:
        result = session.run(cypher)
        return [list(r.values())[0] for r in result]


# === 🌐 Streamlit UI ===
st.title("💊 PrimeKG 問答系統")
question = st.text_input("請輸入問題（例如：哪些藥物可以治療 asthma？）")
llm_model = st.selectbox("選擇 CYPHER 模型", ["llama3", "mistral","tinyllama", "phi3", "gemma:2b"])
llm_model2 = st.selectbox("選擇 LLM 模型", ["llama3", "mistral","tinyllama", "phi3", "gemma:2b"])

if st.button("🔍 查詢") and question:
    with st.spinner("LLM 正在生成查詢語句..."):
        cypher_raw = generate_cypher_query(question, model=llm_model)
        cypher_cleaned = clean_cypher_output(cypher_raw)
        cypher_query = sanitize_cypher(cypher_cleaned)

    st.code(cypher_query, language="cypher")

    try:
        result = run_cypher_query(cypher_query)
        st.success("查詢成功！")
        st.write("📊 結果：", result)

        answer = generate_answer_with_model(question, result, model=llm_model2)
        st.markdown("🗣️ **自然語言回覆：**")
        st.write(answer)

    except Exception as e:
        st.error(f"⚠️ 發生錯誤：{e}")
