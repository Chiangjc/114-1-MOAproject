## STEP 1: Start Neo4j on Docker

``` bash
docker run -it --rm \
 --name primekg \
 --publish=7474:7474 --publish=7687:7687 \
 --env NEO4J_AUTH=neo4j/<password> \
 --env NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
 --env NEO4J_PLUGINS='["graph-data-science"]' \
 -v <隨便一個可以放 data 的地方，應該最好是跟放 primekg dataset 的地方一樣>/neo4j_data:/data \
 -v <你放 primekg dataset 的地方>:/var/lib/neo4j/import \
 neo4j:5.19-enterprise
```

## STEP 2: Load csv on Neo4j

```bash
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CREATE (n:PrimeKGNode {node_index: toInteger(row.node_index)})
SET n.name = row.node_name
RETURN count(n);
```
```bash
LOAD CSV WITH HEADERS FROM 'file:///drug_features.csv' AS row
MERGE (dr:Drug {node_index: toInteger(row.node_index)}) // 使用 MERGE 以免重複創建
SET dr.mechanism_of_action = row.mechanism_of_action,
dr.indication = row.indication 
RETURN count(dr);
```

## STEP 3: Do embedding using Python
因為 disease_features.csv 感覺比較有用的欄位丟進 Neo4j 之後一直出問題，好像資料本身沒有處理乾淨，所以只有先弄 drug_features.csv 的 embedding 比較省時間，之後再處理 disease_features.csv
```bash
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm # 用於顯示進度條

# --- Neo4j 連線設定 ---
# 請替換為您的 Neo4j 連線資訊
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "" # *** 請務必替換為您的 Neo4j 密碼 ***
NEO4J_DATABASE = "neo4j" # 您的資料庫名稱，通常是 'neo4j'

# --- 檔案路徑設定 ---
DISEASE_FEATURES_PATH = '<放disease_features.csv的地方>'
DRUG_FEATURES_PATH = '<放drug_features.csv的地方>'

# --- 嵌入模型設定 ---
# 選擇一個適合的模型。對於醫療/生物領域，可能考慮 specialized 的模型，
# 但 'all-MiniLM-L6-v2' 是一個很好的通用起始點，速度快且效果不錯。
# 如果需要更好的效果，可以考慮 'all-mpnet-base-v2' 或 'BAAI/bge-small-en-v1.5'
# 對於醫學領域，可能考慮像 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
# 但這需要額外的處理步驟來獲取嵌入，並且執行速度會慢很多。
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()

print(f"使用的嵌入模型：{EMBEDDING_MODEL_NAME}")
print(f"嵌入向量維度：{EMBEDDING_DIMENSION}")

# --- Neo4j 驅動初始化 ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_session():
    return driver.session(database=NEO4J_DATABASE)

# --- 函數：從 CSV 讀取資料並生成嵌入 ---
def generate_embeddings_from_csv(file_path, id_col, text_cols, label):
    """
    從 CSV 讀取資料，拼接指定文本欄位，生成嵌入向量。
    Args:
        file_path (str): CSV 檔案路徑。
        id_col (str): 作為唯一識別符的欄位名稱 (例如 'node_index', 'mondo_id')。
        text_cols (list): 包含用於生成嵌入的文本欄位名稱列表。
        label (str): 節點在 Neo4j 中的標籤 (例如 'Disease', 'Drug')。
    Returns:
        list: 包含字典的列表，每個字典包含 'id', 'label', 'embedding' 和 'original_id_col_value'。
    """
    print(f"正在處理 {file_path} 中的 {label} 數據...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"錯誤：檔案 '{file_path}' 不存在。")
        return []

    data_with_embeddings = []
    # 使用 tqdm 顯示進度條
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"生成 {label} 嵌入"):
        unique_id = row[id_col] # 獲取用來識別節點的 ID (例如 node_index, mondo_id)

        # 拼接所有指定的文本欄位
        combined_text_parts = []
        for col in text_cols:
            text = str(row[col]).strip()
            if text and text != 'nan': # 檢查文本是否為空或 NaN
                combined_text_parts.append(text)

        if combined_text_parts:
            # 使用句號和空格連接文本，確保語義完整性
            text_to_embed = ". ".join(combined_text_parts)
            embedding = embedding_model.encode(text_to_embed).tolist() # 轉換為列表以存入 Neo4j
            data_with_embeddings.append({
                "id": unique_id, # 使用 node_index 或 mondo_id 作為數據中的 id
                "label": label,
                "embedding": embedding,
                "original_id_col_value": unique_id # 儲存原始 ID 欄位值以便後續匹配
            })
    return data_with_embeddings

# --- 函數：更新 Neo4j 節點的嵌入 ---
def update_node_embeddings_in_neo4j(session, nodes_data, id_property_name):
    """
    將生成的嵌入向量寫入 Neo4j 中對應的節點。
    Args:
        session (neo4j.Session): Neo4j 資料庫會話。
        nodes_data (list): 包含節點數據和嵌入向量的列表。
        id_property_name (str): Neo4j 中用於匹配節點的屬性名稱 (例如 'node_index', 'mondo_id')。
    """
    if not nodes_data:
        print("沒有數據可以更新。")
        return

    # Cypher 查詢：根據 ID 匹配節點並設定 embedding 屬性
    query = f"""
    UNWIND $data AS item
    MATCH (n) WHERE n.{id_property_name} = item.id
    SET n.embedding = item.embedding
    """
    # 這裡我們一次性發送多個批次，每次處理 1000 個節點
    batch_size = 1000
    for i in tqdm(range(0, len(nodes_data), batch_size), desc=f"更新 Neo4j 節點嵌入 (依據 {id_property_name})"):
        batch = nodes_data[i:i + batch_size]
        session.write_transaction(lambda tx, b: tx.run(query, data=[{"id": item["id"], "embedding": item["embedding"]} for item in b]), batch)
    print(f"已成功更新 {len(nodes_data)} 個節點的嵌入。")

# --- 主程式流程 ---
if __name__ == "__main__":
    try:
        with get_session() as session:
            # --- 處理 Disease 節點 ---
            # 假設 Disease 節點在 Neo4j 中有一個 'mondo_id' 屬性與 CSV 中的 'mondo_id' 對應
            # 這裡我們使用 CSV 的 node_index 作為 Neo4j 節點的 node_index 來匹配，
            # 如果您的 Neo4j 節點是用 mondo_id 識別的，請將 'node_index' 改為 'mondo_id'
            # disease_embeddings = generate_embeddings_from_csv(
            #     DISEASE_FEATURES_PATH,
            #     id_col='node_index', # 或 'mondo_id'，取決於您的 Neo4j 節點實際屬性
            #     text_cols=['mondo_definition'],
            #     label='Disease'
            # )
            # update_node_embeddings_in_neo4j(session, disease_embeddings, 'node_index') # 匹配 Neo4j 節點的屬性

            # --- 處理 Drug 節點 ---
            # 假設 Drug 節點在 Neo4j 中有一個 'node_index' 屬性與 CSV 中的 'node_index' 對應
            drug_embeddings = generate_embeddings_from_csv(
                DRUG_FEATURES_PATH,
                id_col='node_index',
                text_cols=['mechanism_of_action', 'indication'], # 拼接兩個欄位
                label='Drug'
            )
            update_node_embeddings_in_neo4j(session, drug_embeddings, 'node_index')

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            driver.close()
            print("Neo4j 驅動已關閉。")
```

## STEP 4: Check embedding nodes on Neo4j

```bash
MATCH (d:Drug)
WHERE d.embedding IS NOT NULL
RETURN d.node_index, d.embedding 
LIMIT 5;
```

## STEP 5: Create vector index on Neo4j 

```bash
CREATE VECTOR INDEX drug_embedding_index
FOR (d:Drug) ON (d.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
};
```

## STEP 6: Search for similar nodes (cosine similarity)

```bash
MATCH (q:Drug {node_index: 14022}) // 範例：找與節點 id 為 14022 的相似節點
WITH q.embedding AS queryVec
CALL db.index.vector.queryNodes(
  'drug_embedding_index',  
  5,                      
  queryVec                
) YIELD node AS similarDrug, score
RETURN similarDrug.node_index AS id, score
ORDER BY score DESC;
```