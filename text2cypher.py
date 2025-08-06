from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
from tabulate import tabulate


# 連接 Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# 初始化本地 Ollama 模型
llm = OllamaLLM(model_name="llama3", model_params={"temperature": 0})

# 初始化 Retriever
retriever = Text2CypherRetriever(driver=driver, llm=llm)

# 查詢語句
query = "What drugs are related to asthma?"

# 拿到結果
results = retriever.search(query_text=query)

# 顯示每個結果
for idx, (cypher_query, records) in enumerate(results):
    print(f"\n=== Result {idx + 1} ===")
    print("Generated Cypher:\n", cypher_query)

    if records:
        rows = []
        for record in records:
            try:
                # 嘗試將 record 轉成 dict（可使用 record.data() 或 dict(record)）
                row = dict(record)
                rows.append(row)
            except Exception as e:
                print(f"Error parsing record: {e}")
                print("Raw record:", record)

        if rows:
            print(tabulate(rows, headers="keys", tablefmt="pretty"))
        else:
            print("No rows found after parsing.")
    else:
        print("No records found.")

# 執行 GraphRAG 對話生成
rag = GraphRAG(retriever=retriever, llm=llm)
answer = rag.search(query_text=query)
print("\nRAG generated answer:\n", answer.answer)
