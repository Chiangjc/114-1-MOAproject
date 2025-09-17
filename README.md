# Neo4j 5.20 + APOC 安裝指南

## 1. 下載 APOC jar
前往官方 release 頁面：[APOC Releases](https://github.com/neo4j/apoc/releases)  
下載與 Neo4j 5.20 相容的 jar，例如：`apoc-5.20.0-core.jar`  
建立 `plugins` 資料夾並將檔案放入。

## 2. 啟動 Neo4j Docker Container
使用以下指令啟動 Neo4j Docker 容器：

```powershell
docker run -d --name neo4j `
    -p 7474:7474 -p 7687:7687 `
    -e NEO4J_AUTH=neo4j/test1234 `
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.* `
    -e NEO4J_dbms_security_procedures_allowlist=apoc.* `
    -v path\to\data:/data `
    -v path\to\plugins:/var/lib/neo4j/plugins `
    neo4j:5.20
```

## 3. 安裝 Python 依賴
執行以下指令安裝 Python 依賴：

```bash
pip install -r requirements.txt
```

## 4. 啟動 Open-WebUI Docker
使用以下指令啟動 Open-WebUI Docker 容器：

```bash
docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda
```

## 5. 啟動 Ollama Docker
使用以下指令啟動 Ollama Docker 容器：

```bash
docker run -d -p 11434:11434 ollama/ollama:latest
```

## 6. 啟動 Streamlit 應用
執行以下指令啟動 Streamlit 應用：

```bash
streamlit run text2cypher.py
```