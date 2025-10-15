import streamlit as st
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
import pandas as pd
import re
import ast
import os
import json
from Bio import Entrez
import requests
from bs4 import BeautifulSoup
import asyncio
import time  # ▶︎ 用於計時
import hashlib
from datetime import datetime, timedelta
import torch, gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def safe_load_sentence_transformer(model_name: str, device: str = "cpu", max_retries: int = 3):
    """Safely load SentenceTransformer, handling meta tensor errors."""
    model = None
    for attempt in range(1, max_retries + 1):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)

            model = SentenceTransformer(model_name, device=device)
            _ = model.encode("test", convert_to_numpy=True)
            return model

        except RuntimeError as e:
            if "meta tensor" in str(e).lower():
                st.warning(f"⚠️ Meta tensor error — retrying load (attempt {attempt}/{max_retries})...")
                if model is not None:
                    del model
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(2)
            else:
                raise e
    raise RuntimeError("Failed to load model after 3 retries due to meta tensor errors.")


# ====================== 新增：Qdrant 緩存 ======================
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ====================== 可調參數 ======================
TOP_N_PAPERS = 5  # ▶︎ PapersRAG 最多引用的論文篇數（可自行改成 10 等）
QDRANT_PATH = "./qdrant_cache"  # ▶︎ 本地端 Qdrant 儲存路徑（RocksDB）
QDRANT_COLLECTION = "papers_chunks_v1"  # ▶︎ 儲存論文切塊

# ========== Streamlit UI ==========
st.set_page_config(page_title="Medical RAG System", layout="wide")
st.title("🧠 Multi-Agent Medical RAG System")
st.markdown("結合 **GraphRAG (PrimeKG)** + **PubMed Papers RAG**，並使用 **Qdrant** 本地快取論文片段")

query = st.text_input("請輸入您的醫學問題")

managing_model = st.selectbox(
    "選擇 Managing Agent 模型",
    ["llama3", "mistral", "qwen2:7b", "gpt-oss"]
)
cypher_model = st.selectbox(
    "選擇 Cypher 查詢模型",
    ["gpt-oss"]
)
answer_model = st.selectbox(
    "選擇 回答生成模型",
    ["llama3", "mistral", "gpt-oss"]
)

# ✅ 改成 Qdrant 緩存控制（優先權：僅用緩存 > 強制刷新）
cache_only_search = st.checkbox("僅在 Qdrant 緩存中查詢 (跳過 PubMed 搜索/下載)", value=False)
force_pubmed_refresh = st.checkbox("強制重新查詢 PubMed 並更新緩存", value=False, disabled=cache_only_search)

submit = st.button("查詢")

# ========== Neo4j Initialization ==========
@st.cache_resource
def init_graph(cypher_model_name, answer_model_name):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    cypher_llm = OllamaLLM(model_name=cypher_model_name, model_params={"temperature": 0}, host="http://localhost:11434")
    answer_llm = Ollama(model=answer_model_name, temperature=0, base_url="http://localhost:11434")
    retriever = Text2CypherRetriever(driver=driver, llm=cypher_llm)
    return driver, cypher_llm, answer_llm, retriever

# ========== Qdrant Initialization ==========
@st.cache_resource
def init_qdrant(emb_dim: int):
    client = QdrantClient(path=QDRANT_PATH)  # 本地 RocksDB，免啟動 Server
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE),
        )
    return client

# ========== Embedding / Reranker ==========
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _emb_model = safe_load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
    EMB_DIM = _emb_model.get_sentence_embedding_dimension()
    _reranker = None
    try:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _reranker = None
except Exception:
    _emb_model = None
    EMB_DIM = 384
    _reranker = None

# ========== Start drivers ==========
driver, cypher_llm, answer_llm, retriever = init_graph(cypher_model, answer_model)
qdrant = init_qdrant(EMB_DIM)

# ========== PubMed Utility Functions ==========
Entrez.email = "email@gmail.com"  # TODO: Replace with your email

def search_pubmed(keyword, max_results=20):
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    return record["IdList"]

def get_pmc_id(pmid):
    handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
    record = Entrez.read(handle)
    try:
        return record[0]['LinkSetDb'][0]['Link'][0]['Id']
    except:
        return None

def get_title_from_pubmed(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    lines = handle.read().split("\n")
    title_lines = [line[6:].strip() for line in lines if line.startswith("TI  ") or line.startswith("      ")]
    return " ".join(title_lines) if title_lines else ""

def get_fulltext_from_europe_pmc(pmcid):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{pmcid}/fullTextXML"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def extract_fulltext_structure(xml_text):
    soup = BeautifulSoup(xml_text, "xml")
    title = soup.find("article-title")
    abstract = soup.find("abstract")
    body_paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
    return {
        "title": title.get_text().strip() if title else "",
        "abstract": abstract.get_text(separator=" ", strip=True) if abstract else "",
        "body": body_paragraphs
    }

def retrieve_papers(keyword, max_results=20):
    pmids = search_pubmed(keyword, max_results=max_results)
    papers = []
    for pmid in pmids:
        pmc_id = get_pmc_id(pmid)
        title = get_title_from_pubmed(pmid)
        entry = {
            "pmid": pmid,
            "title": title,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }
        if pmc_id:
            xml = get_fulltext_from_europe_pmc(pmc_id)
            if xml:
                entry["fulltext"] = extract_fulltext_structure(xml)
        papers.append(entry)
    return papers

# ========== Basic text helpers ==========
_STOPWORDS = set("""
the a an and or of for to with without in on by from about as is are was were be being been into over under than then this that those these such using use used based among between
""".split())

def _clean_phrase(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s

def derive_core_terms(user_query: str) -> list[str]:
    q = user_query.strip()
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', q)
    phrases = [p[0] or p[1] for p in quoted]
    if phrases:
        core = [_clean_phrase(p) for p in phrases if _clean_phrase(p)]
    else:
        toks = [t for t in re.split(r"[^A-Za-z0-9\-]+", q) if t]
        toks = [t for t in toks if len(t) >= 3 and t.lower() not in _STOPWORDS]
        core = toks[:5] if toks else [q]
    seen, core_unique = set(), []
    for t in core:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            core_unique.append(t)
    return core_unique

def contains_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms)

# ========== RAG chunking ==========
import unicodedata

def _normalize_ws(s: str) -> str:
    return " ".join(s.split())

def _split_sentences(text: str) -> list[str]:
    import re as _re
    text = text.strip()
    if not text:
        return []
    sentences = _re.split(r'(?<=[。！？!?\.])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def build_chunks_from_entry(entry: dict):
    chunks = []
    pmid = entry.get("pmid")
    title = entry.get("title", "")
    url = entry.get("pubmed_url", "")

    # 摘要
    if entry.get("fulltext", {}).get("abstract"):
        for i, ch in enumerate(_split_sentences(entry["fulltext"]["abstract"])):
            chunks.append({
                "pmid": pmid, "title": title, "where": "abstract",
                "chunk_id": f"abs-s{i}", "text": ch, "pubmed_url": url
            })

    # 正文
    if entry.get("fulltext", {}).get("body"):
        for j, para in enumerate(entry["fulltext"]["body"]):
            for i, ch in enumerate(_split_sentences(para)):
                chunks.append({
                    "pmid": pmid, "title": title, "where": "body",
                    "chunk_id": f"p{j}-s{i}", "text": ch, "pubmed_url": url
                })

    # 沒有全文時，至少用 title
    if not chunks and title.strip():
        chunks.append({
            "pmid": pmid, "title": title, "where": "title",
            "chunk_id": "t-0", "text": title.strip(), "pubmed_url": url
        })

    return chunks

# ========== Qdrant Cache Helpers ==========

def _point_id(pmid: str, chunk_id: str) -> int:
    # Qdrant 支援 64-bit 整數或字串 id；用 md5 轉成 64-bit 整數（取前 16 位）
    h = hashlib.md5(f"{pmid}:{chunk_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def qdrant_upsert_papers(client: QdrantClient, papers: list[dict]):
    if _emb_model is None:
        st.warning("⚠️ sentence-transformers 未安裝，無法建立 Qdrant 緩存。")
        return 0, 0

    points = []
    total_chunks = 0
    for p in papers:
        chunks = build_chunks_from_entry(p)
        total_chunks += len(chunks)
        if not chunks:
            continue
        texts = [c["text"] for c in chunks]
        vecs = _emb_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        for c, v in zip(chunks, vecs):
            pid = _point_id(c.get("pmid", "NA"), c.get("chunk_id", "NA"))
            payload = {**c}
            points.append(PointStruct(id=pid, vector=v.tolist(), payload=payload))

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(papers), total_chunks


def qdrant_search_chunks(client: QdrantClient, query: str, top_k: int = 64):
    if _emb_model is None:
        return []
    qvec = _emb_model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0].tolist()
    res = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
    hits = []
    for r in res:
        payload = r.payload or {}
        hits.append({
            "text": payload.get("text", ""),
            "meta": {
                "pmid": payload.get("pmid"),
                "title": payload.get("title"),
                "where": payload.get("where"),
                "pubmed_url": payload.get("pubmed_url"),
                "chunk_id": payload.get("chunk_id"),
            },
            "score": float(r.score),
        })
    return hits


def rerank_hits(query: str, hits: list[dict], top_k: int = 8):
    if not hits:
        return []
    if _reranker is None:
        return hits[:top_k]
    pairs = [(query, h["text"]) for h in hits[: max(32, top_k*2)]]
    scores = _reranker.predict(pairs)
    for h, s in zip(hits[:len(pairs)], scores):
        h["rerank"] = float(s)
    hits = sorted(hits[:len(pairs)], key=lambda x: x.get("rerank", 0.0), reverse=True)[:top_k]
    return hits


def make_cited_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        pmid = m.get("pmid", "NA")
        title = m.get("title", "")
        where = m.get("where", "")
        url = m.get("pubmed_url", "")
        txt = h["text"]
        blocks.append(
            f"[{i}] PMID {pmid} | {title} | {where}\n{txt}\nLink: {url}"
        )
    return "\n\n".join(blocks)


def filter_hits_by_terms(hits: list[dict], must_terms: list[str]) -> list[dict]:
    if not must_terms:
        return hits
    out = []
    for h in hits:
        m = h.get("meta", {})
        if contains_any(h.get("text", ""), must_terms) or contains_any(m.get("title", ""), must_terms):
            out.append(h)
    return out

# ========== Final Answer Synthesizer ==========
def synthesize_answer(query, graph_context, papers_context, model):
    llm = Ollama(model=model, temperature=0, base_url="http://localhost:11434")
    prompt = PromptTemplate(
        input_variables=["question", "graph", "papers"],
        template="""
你是一個醫學助理，請根據以下來源產生回答。務必遵守以下規則：

1. 僅能使用來源1 (GraphRAG) 和來源2 (PapersRAG) 提供的內容，不要自行新增任何額外知識。
2. 整理來源中的有用資訊，避免重複，但要盡可能涵蓋所有正確的細節。
3. 如果引用 來源1，請標註「[GraphRAG]」，並只用節點與關係資訊。
4. 如果引用 來源2，請標註 PMID，並直接引用來源中出現的段落文字或數據。
   - 格式範例：根據 PMID: 123456 的研究，藥物 X 可改善氣喘控制。
   - 不要生成或修改來源中未出現的文獻。
5. 若同時有兩個來源，請先整理 GraphRAG 的結構化知識，再整理 PapersRAG 的研究結果，最後做簡短綜合。
6. 保持專業、客觀、精簡，避免誇大或臆測。

---

來源1 (GraphRAG 知識圖譜):
{graph}

來源2 (PapersRAG 最新醫學文獻):
{papers}

問題：
{question}

請依規則，整理出最終回答。
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question=query, graph=graph_context, papers=papers_context)

# ========== Multi-Agent Managing System ==========
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class SubQuery(BaseModel):
    subquery: str = Field(description="The rewritten sub-question")
    agent: str = Field(description="Which agent should handle this sub-question. Must be either GraphRAG or PapersRAG")

class QueryPlan(BaseModel):
    tasks: List[SubQuery]

def create_query_plan(query, llm_model):
    llm = Ollama(model=llm_model, temperature=0, base_url="http://localhost:11434")
    parser = PydanticOutputParser(pydantic_object=QueryPlan)
    prompt = PromptTemplate(
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
你是一個醫學多代理系統的任務管理代理，負責將複雜的醫學問題拆解為子問題，並指派給不同的專家代理：

- GraphRAG Agent: 適合查詢結構化生物醫學知識圖譜。
- PapersRAG Agent: 適合查詢最新醫學文獻，請特別注意指派給他的問題只能是英文，如果是中文或其他語言的話，請先翻譯成英文再指派。

對以下問題進行分析，產生一個 JSON 格式的任務計畫，格式必須為：
{format_instructions}

問題：
{question}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(question=query)
    return parser.parse(output)

# ===== Main Logic =====
if submit and query:
    overall_t0 = time.perf_counter()
    with st.spinner("正在進行多代理協作查詢..."):
        try:
            # Step 1. Create task plan（計時）
            tm_t0 = time.perf_counter()
            plan = create_query_plan(query, managing_model)
            tm_t1 = time.perf_counter()

            st.info("🤖 Managing Agent 建立的查詢計畫：")
            st.json(plan.model_dump())
            st.caption(f"⏱️ Managing Agent 用時：{tm_t1 - tm_t0:.2f} s")

            graph_context = ""  # ← 初始化 GraphRAG 上下文
            papers_context = "" # ← 初始化 PapersRAG 上下文

            # Step 2. Execute subtasks
            for task in plan.tasks:
                st.write(f"🔹 處理子問題: **{task.subquery}** → 指派給 **{task.agent}**")

                if task.agent == "GraphRAG":
                    gr_t0 = time.perf_counter()
                    st.subheader("🧬 GraphRAG 子任務")

                    # ==================================================
                    # 1️⃣ 問題類型辨識
                    # ==================================================
                    def detect_question_type(question: str) -> str:
                        opt_like = re.search(r"(\([A-D]\)|\d+\.)", question)
                        yn_like = re.search(r"\b(是|否|可能|yes|no|true|false)\b", question, re.I)
                        if opt_like and not yn_like:
                            return "選項感知"
                        else:
                            return "非選項感知"

                    qtype = detect_question_type(task.subquery)
                    st.caption(f"🧩 問題類型：{qtype}")

                    # ==================================================
                    # 2️⃣ Triplet 生成階段
                    # ==================================================
                    from langchain.llms import Ollama as _Ollama
                    triplet_t0 = time.perf_counter()

                    llm_gen = _Ollama(model=cypher_model, temperature=0, base_url="http://localhost:11434")

                    if qtype == "選項感知":
                        prompt = f"""
                你是一個醫學知識圖譜構建專家。
                請將以下「選項感知型」問題分解成多組三元組（triplets）。
                請逐一考慮每個選項的內容（如疾病、基因、藥物、症狀），避免順序偏差。
                請直接輸出 JSON，例如：
                [
                {{ "head": "DrugA", "relation": "treats", "tail": "DiseaseX" }},
                {{ "head": "GeneB", "relation": "associated_with", "tail": "DiseaseY" }}
                ]

                問題：
                {task.subquery}
                """
                    else:
                        prompt = f"""
                你是一個醫學知識圖譜推理專家。
                請根據以下問題，提取出可在生醫知識圖譜中表示的三元組（triplets）。
                問題屬於非選項感知型（如是/否問題），只根據問題主體中的醫療概念生成三元組。
                請直接輸出 JSON，例如：
                [
                {{ "head": "GeneX", "relation": "associated_with", "tail": "DiseaseY" }}
                ]

                問題：
                {task.subquery}
                """

                    st.write("🧠 由 LLM 生成可能的三元組...")
                    raw_output = llm_gen.invoke(prompt)
                    try:
                        triplets = json.loads(raw_output)
                        if isinstance(triplets, dict):
                            triplets = [triplets]
                    except Exception:
                        triplets = [{"raw_output": raw_output}]
                    st.json(triplets)
                    triplet_t1 = time.perf_counter()
                    st.caption(f"⏱️ Triplet 生成耗時：{triplet_t1 - triplet_t0:.2f} s")

                    # ==================================================
                    # 3️⃣ 與 Neo4j 知識圖譜驗證階段
                    # ==================================================
                    def verify_triplets_with_kg(triplets: list[dict], driver):
                        verified = []
                        with driver.session() as session:
                            for t in triplets:
                                if not all(k in t for k in ["head", "relation", "tail"]):
                                    continue
                                cypher = f"""
                                MATCH (h)-[r]->(t)
                                WHERE toLower(h.name) CONTAINS toLower($head)
                                AND toLower(t.name) CONTAINS toLower($tail)
                                AND toLower(type(r)) CONTAINS toLower($rel)
                                RETURN h.name AS head, type(r) AS relation, t.name AS tail
                                LIMIT 1
                                """
                                res = session.run(cypher, head=t["head"], tail=t["tail"], rel=t["relation"]).data()
                                verified.append({
                                    **t,
                                    "exists_in_KG": bool(res),
                                    "matched": res[0] if res else None
                                })
                        return verified

                    st.write("🔍 與 Neo4j 知識圖譜比對中...")
                    verify_t0 = time.perf_counter()
                    verified = verify_triplets_with_kg(triplets, driver)
                    verify_t1 = time.perf_counter()
                    st.json(verified)
                    st.caption(f"⏱️ Neo4j 驗證耗時：{verify_t1 - verify_t0:.2f} s")

                    # ==================================================
                    # 4️⃣ LLM 裁判代理（Triplet 合理性判斷）
                    # ==================================================
                    st.write("⚖️ 由裁判代理判斷 triplets 是否合理...")
                    judge_t0 = time.perf_counter()

                    llm_judge = Ollama(model="gpt-oss", temperature=0, base_url="http://localhost:11434")
                    triplet_text = json.dumps(verified, ensure_ascii=False, indent=2)

                    judge_prompt = f"""
                你是一個熟悉生醫知識圖譜結構的判斷代理。
                請根據以下 triplets 判斷其是否符合醫學常識與圖譜結構。
                請輸出 JSON 格式：
                [
                {{ "head": "...", "relation": "...", "tail": "...", "is_valid": true/false, "comment": "..." }}
                ]

                以下是待檢查的 triplets：
                {triplet_text}
                """

                    judge_output = llm_judge.invoke(judge_prompt)
                    try:
                        judged = json.loads(judge_output)
                        if isinstance(judged, dict):
                            judged = [judged]
                    except Exception:
                        judged = [{"raw_output": judge_output}]
                    st.json(judged)
                    judge_t1 = time.perf_counter()
                    st.caption(f"⏱️ 裁判代理耗時：{judge_t1 - judge_t0:.2f} s")

                    # ==================================================
                    # 5️⃣ 累積通過檢核的 triplets
                    # ==================================================
                    valid_tris = [j for j in judged if j.get("is_valid", True)]
                    if valid_tris:
                        st.success(f"✅ 通過檢核 triplets: {len(valid_tris)} 條")
                        graph_context += "\n".join([json.dumps(t, ensure_ascii=False) for t in valid_tris]) + "\n"
                    else:
                        st.warning("⚠️ 無通過檢核的 triplets，略過。")

                    st.subheader("📊 通過驗證的結構化路徑")
                    for t in valid_tris:
                        head = t.get("head", "?")
                        rel = t.get("relation", "?")
                        tail = t.get("tail", "?")
                        st.markdown(f"- **{head}** —[{rel}]→ **{tail}**")

                    # （可選）Embedding-based 查詢：維持原樣
                    try:
                        device = "cpu"
                        st.caption(f"🔍 Embedding 檢索裝置：{device}")
                        embedding_model = safe_load_sentence_transformer("all-MiniLM-L6-v2", device=device, max_retries=3)
                        
                        query_vec = embedding_model.encode(
                            task.subquery,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        ).tolist()
                        with driver.session() as session:
                            result = session.run(
                                """
                                WITH $queryVec AS queryVec
                                CALL db.index.vector.queryNodes(
                                    'drug_embedding_index', 5, queryVec
                                ) YIELD node, score
                                RETURN node.node_index AS id, node.indication AS indication, score
                                ORDER BY score DESC
                                """,
                                queryVec=query_vec,
                            )
                            embed_records = [dict(record) for record in result]
                        if embed_records:
                            st.subheader("📊 GraphRAG Embedding 查詢結果")
                            df_embed = pd.DataFrame(embed_records)
                            st.dataframe(df_embed)
                            graph_context += "\n".join([str(r) for r in embed_records]) + "\n"
                        
                        if embed_records:
                            st.subheader("🧠 Subgraph Expansion from Top Drugs")
                            retrieved_ids = [str(r["id"]) for r in embed_records if "id" in r]
                            if retrieved_ids:
                                with driver.session() as session:
                                    subgraph_query = """
                                    MATCH path=(d:drug)-[*1..2]-(n)
                                    WHERE d.node_index IN $retrieved_ids
                                    RETURN d.node_name AS drug,
                                        reduce(s = '', r IN relationships(path) |
                                                s + startNode(r).node_name + ' —[' + type(r) + ']→ ' + endNode(r).node_name + '; ') AS path
                                    LIMIT 50
                                    """
                                    paths = session.run(subgraph_query, parameters={"retrieved_ids": retrieved_ids}).data()

                                if paths:
                                    st.dataframe(pd.DataFrame(paths))
                                    graph_context += "\n".join([p["path"] for p in paths]) + "\n"
                                    # ===== Streamlit 視覺化 =====
                                    import networkx as nx
                                    import matplotlib.pyplot as plt
                                    G = nx.DiGraph()
                                    for p in paths:
                                        edges = p["path"].split("; ")
                                        for e in edges:
                                            if not e.strip():
                                                continue
                                            try:
                                                src, rest = e.split(" —[")
                                                rel, tgt = rest.split("]→ ")
                                                G.add_edge(src.strip(), tgt.strip(), relation=rel.strip())
                                            except ValueError:
                                                continue  # 忽略解析失敗的邊
                                    # 畫圖
                                    plt.figure(figsize=(12, 8))
                                    pos = nx.spring_layout(G, seed=42)
                                    edge_labels = nx.get_edge_attributes(G, "relation")
                                    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10)
                                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
                                    st.pyplot(plt)
                                else:
                                    st.warning("No subgraph paths found for retrieved drug IDs.")

                    except Exception as e:
                        st.warning(f"⚠️ Embedding 檢索失敗: {e}")

                    gr_t1 = time.perf_counter()
                    st.caption(f"⏱️ GraphRAG 子任務總耗時：{gr_t1 - gr_t0:.2f} s")

                elif task.agent == "PapersRAG":
                    pr_t0 = time.perf_counter()

                    # A) must terms（僅用於過濾）
                    must_terms = derive_core_terms(task.subquery)

                    selected_hits = []
                    selected_set = set()

                    # 先嘗試：若非強制刷新，優先查 Qdrant 緩存
                    used_cache = False
                    if not force_pubmed_refresh:
                        cache_hits = qdrant_search_chunks(qdrant, task.subquery, top_k=128)
                        cache_hits = filter_hits_by_terms(cache_hits, must_terms)
                        if cache_hits:
                            st.caption(f"📦 來自 Qdrant 緩存的命中片段：{len(cache_hits)}")
                            # 將緩存命中做簡單精排與前 N 論文抽取
                            cache_hits = rerank_hits(task.subquery, cache_hits, top_k=64)
                            ordered_pmids = []
                            for h in cache_hits:
                                pmid = h["meta"].get("pmid")
                                if pmid and pmid not in ordered_pmids:
                                    ordered_pmids.append(pmid)
                                if len(ordered_pmids) >= int(TOP_N_PAPERS):
                                    break
                            selected_set = set(ordered_pmids)
                            selected_hits = [h for h in cache_hits if h["meta"].get("pmid") in selected_set]
                            used_cache = True

                    # 若沒有命中或使用者要求刷新，查 PubMed 並更新緩存
                    if (not used_cache and not cache_only_search) or force_pubmed_refresh:
                        t_pubmed_0 = time.perf_counter()
                        raw_query = task.subquery
                        papers = retrieve_papers(raw_query, max_results=20)
                        t_pubmed_1 = time.perf_counter()
                        st.caption(f"⏱️ PubMed 檢索：{t_pubmed_1 - t_pubmed_0:.2f} s，取得 {len(papers)} 篇")

                        # 把最相關的 20 篇寫進 Qdrant 緩存
                        up_n, up_chunks = qdrant_upsert_papers(qdrant, papers)
                        st.caption(f"💾 已緩存到 Qdrant：論文 {up_n} 篇、片段 {up_chunks} 條")

                        # 再次到緩存取片段
                        cache_hits = qdrant_search_chunks(qdrant, task.subquery, top_k=128)
                        cache_hits = filter_hits_by_terms(cache_hits, must_terms)
                        cache_hits = rerank_hits(task.subquery, cache_hits, top_k=64)

                        ordered_pmids = []
                        for h in cache_hits:
                            pmid = h["meta"].get("pmid")
                            if pmid and pmid not in ordered_pmids:
                                ordered_pmids.append(pmid)
                            if len(ordered_pmids) >= int(TOP_N_PAPERS):
                                break
                        selected_set = set(ordered_pmids)
                        selected_hits = [h for h in cache_hits if h["meta"].get("pmid") in selected_set]

                    # 呈現結果 & 組裝 context
                    if not selected_hits:
                        st.subheader("📄 PapersRAG 子查詢結果")
                        if cache_only_search:
                            st.info("Qdrant 緩存未命中，且目前設定為僅查緩存，因此未呼叫 PubMed。")
                        else:
                            st.info("沒有合適的片段。可嘗試變更查詢或勾選『強制重新查詢 PubMed』。")
                        papers_context += (
                            f"\n### 子問題: {task.subquery}\n(未取得可用片段)\n"
                        )
                    else:
                        t_ctx_0 = time.perf_counter()
                        cited_context = make_cited_context(selected_hits)
                        t_ctx_1 = time.perf_counter()
                        st.caption(f"⏱️ 引文組裝：{t_ctx_1 - t_ctx_0:.2f} s")

                        st.subheader(f"📄 PapersRAG 子查詢結果（取前 {int(TOP_N_PAPERS)} 篇）")
                        for i, h in enumerate(selected_hits, 1):
                            m = h["meta"]
                            pmid = m.get("pmid", "NA")
                            where = m.get("where", "")
                            url = m.get("pubmed_url", "")
                            title = m.get("title", "(no title)")
                            score = h.get("rerank", h.get("score", 0.0))
                            st.markdown(
                                f"**[{i}] {title}** — *{where}*  \n"
                                f"PMID: `{pmid}` ｜ Score: `{score:.3f}` ｜ [PubMed]({url})"
                            )
                            st.write(h["text"][:500] + ("..." if len(h["text"]) > 500 else ""))

                        papers_context += f"\n### 子問題: {task.subquery}\n" + cited_context

                    pr_t1 = time.perf_counter()
                    st.caption(f"⏱️ PapersRAG 子任務用時：{pr_t1 - pr_t0:.2f} s")

            # Step 3. Final synthesis（計時）
            t_synth_0 = time.perf_counter()
            final_answer = synthesize_answer(query, graph_context, papers_context, answer_model)
            t_synth_1 = time.perf_counter()
            st.subheader("💬 最終回答")
            st.success(final_answer)
            st.caption(f"⏱️ 最終答案生成：{t_synth_1 - t_synth_0:.2f} s")

            overall_t1 = time.perf_counter()
            st.caption(f"⏱️ 總用時：{overall_t1 - overall_t0:.2f} s")

        except Exception as e:
            st.error(f"❌ 發生錯誤：{e}")  # 若 Cypher 有錯不致於整個流程中止