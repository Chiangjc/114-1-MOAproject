import requests
from neo4j import GraphDatabase

# 設定連線
uri = "bolt://localhost:7687"  # 預設 bolt port
user = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))

def generate_cypher_query(user_question, model="llama3"):
    prompt = f"""
你是一個擅長 Neo4j 的圖資料查詢專家，請根據下列圖結構與使用者提問，產生對應的 Cypher 查詢語句。
請注意只使用已知的節點與關係，並使用正確的屬性名稱（如 node_name）。

【節點類型與屬性】
- anatomy
- disease
- drug
- effect__phenotype
- biological_process
- molecular_function
- cellular_component
- exposure
- pathway
- gene__protein

每個節點都包含以下屬性：
- node_name（例如 "tinea pedis"）
- node_id, node_source, node_index

【關係類型】
- (drug)-[:indication]->(disease)
- (drug)-[:"off-label use"]->(disease)
- (drug)-[:contraindication]->(disease)
- (drug)-[:drug_protein]->(gene__protein)
- (gene__protein)-[:protein_protein]->(gene__protein)
- (drug)-[:drug_drug]->(drug)

請根據以下問題，輸出正確且完整的 Cypher 查詢語句（不需要加解釋）：


使用者問題：
\"\"\"
{user_question}
\"\"\"

請輸出 Cypher 查詢：
"""
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return res.json()["response"]

def generate_answer_with_model(question: str, results: list[str], model: str = "mistral") -> str:
    prompt = f"""
請根據下列資訊，產生一段自然語言的回答，用中文說明這些查詢結果與問題的關聯。

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



def run_cypher_query(cypher):
    with driver.session() as session:
        result = session.run(cypher)
        return [list(r.values())[0] for r in result]

# question = "有哪些藥物可以治療 tinea pedis？"
question = input("please ask any question: ")
cypher_query = generate_cypher_query(question)
print("🔍 生成的 Cypher 查詢：")
print(cypher_query)

result = run_cypher_query(cypher_query)
print("📊 查詢結果：")
print(result)

answer = generate_answer_with_model(question, result, model="mistral")
print("🗣️ 自然語言回覆：")
print(answer)