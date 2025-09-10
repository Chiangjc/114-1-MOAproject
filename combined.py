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
from tqdm import tqdm
import asyncio

# ========== Streamlit UI ==========
st.set_page_config(page_title="Medical RAG System", layout="wide")
st.title("🧠 Multi-Agent Medical RAG System")
st.markdown("結合 **GraphRAG (PrimeKG)** + **PubMed Papers RAG**，為您提供醫學研究答案")

query = st.text_input("請輸入您的醫學問題")

managing_model = st.selectbox(
    "選擇 Managing Agent 模型",
    ["llama3", "mistral", "qwen2:7b", "gpt-oss"]
)
cypher_model = st.selectbox(
    "選擇 Cypher 查詢模型",
    ["llama3", "mistral", "gemma:7b", "qwen2:7b", "gpt-oss"]
)
answer_model = st.selectbox(
    "選擇 回答生成模型",
    ["llama3", "mistral", "gpt-oss"]
)

submit = st.button("查詢")

# ========== Neo4j Initialization ==========
@st.cache_resource
def init_graph(cypher_model_name, answer_model_name):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    cypher_llm = OllamaLLM(model_name=cypher_model_name, model_params={"temperature": 0}, host="http://localhost:11434")
    answer_llm = Ollama(model=answer_model_name, temperature=0, base_url="http://localhost:11434")
    retriever = Text2CypherRetriever(driver=driver, llm=cypher_llm)
    return driver, cypher_llm, answer_llm, retriever

driver, cypher_llm, answer_llm, retriever = init_graph(cypher_model, answer_model)

# ========== PubMed Utility Functions ==========
Entrez.email = "youremail@gmail.com"  # TODO: Replace with your email

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

import re

# 簡單停用字，可自行擴充
_STOPWORDS = set("""
the a an and or of for to with without in on by from about as is are was were be being been into over under than then this that those these such using use used based among between
""".split())

def _clean_phrase(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s

def derive_core_terms(user_query: str) -> list[str]:
    """
    從使用者 query 動態抽主題詞（疾病 / 主體），作為 must terms 與 PubMed 查詢核心。
    規則：
    - 先抓引號中的片語
    - 沒有引號則取長度 >= 3 的關鍵詞，去停用字
    - 至少保留一個詞（整句）
    """
    q = user_query.strip()
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', q)
    phrases = [p[0] or p[1] for p in quoted]
    if phrases:
        core = [_clean_phrase(p) for p in phrases if _clean_phrase(p)]
    else:
        toks = [t for t in re.split(r"[^A-Za-z0-9\-]+", q) if t]
        toks = [t for t in toks if len(t) >= 3 and t.lower() not in _STOPWORDS]
        core = toks[:5] if toks else [q]
    # 去重並保留原順序
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

# ========== PapersRAG RAG Helpers  ==========

# 可選套件：若已安裝會啟用；未安裝則自動降級
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _emb_model = None
    _reranker = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

import math
import unicodedata

def _normalize_ws(s: str) -> str:
    return " ".join(s.split())

def _chunk_text(text: str, max_chars: int = 1800, overlap: int = 200):
    """
    簡單用字元長度切塊（保守估約 1.2–1.6 chars/token）。
    """
    text = _normalize_ws(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def build_chunks_from_entry(entry: dict):
    """
    把單篇論文 entry 轉為 RAG 可用的 chunks。
    來源：
      - fulltext.abstract
      - fulltext.body (每段落)
      - 無全文時，至少用 title 當線索
    """
    chunks = []
    pmid = entry.get("pmid")
    title = entry.get("title", "")
    url = entry.get("pubmed_url", "")

    # 摘要
    if entry.get("fulltext", {}).get("abstract"):
        for i, ch in enumerate(_chunk_text(entry["fulltext"]["abstract"])):
            chunks.append({
                "pmid": pmid, "title": title, "where": "abstract",
                "chunk_id": f"abs-{i}", "text": ch, "pubmed_url": url
            })

    # 正文
    if entry.get("fulltext", {}).get("body"):
        for j, para in enumerate(entry["fulltext"]["body"]):
            para = _normalize_ws(para)
            if not para:
                continue
            # 你原本沒有 section 名稱，這裡直接標 body
            for i, ch in enumerate(_chunk_text(para)):
                chunks.append({
                    "pmid": pmid, "title": title, "where": "body",
                    "chunk_id": f"p{j}-{i}", "text": ch, "pubmed_url": url
                })

    # 沒有全文時，至少用 title
    if not chunks:
        if title.strip():
            chunks.append({
                "pmid": pmid, "title": title, "where": "title",
                "chunk_id": "t-0", "text": title.strip(), "pubmed_url": url
            })

    return chunks

def _tokenize_for_bm25(txt: str):
    # 很鬆的英文 token 化；中文以字元切
    txt = unicodedata.normalize("NFKC", txt.lower())
    # 留下字母數字與空白
    cleaned = []
    for ch in txt:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        else:
            # 中文 / 其他語系：直接視為 token
            if ord(ch) > 127 and not ch.isspace():
                cleaned.append(f" {ch} ")
    return " ".join("".join(cleaned).split()).split()

class PapersMiniIndex:
    """
    非持久化的小型索引。
    - 若裝了 sentence-transformers：用向量相似度
    - 若裝了 rank-bm25：用 BM25
    - 兩者都有：做簡單 hybrid（分數標準化後合併）
    - 全都沒有：用關鍵詞命中分數
    """
    def __init__(self, chunks: list[dict]):
        self.docs = [c["text"] for c in chunks]
        self.metas = chunks
        self._dense = None
        self._bm25 = None

        # Dense embeddings（可選）
        if _emb_model is not None and self.docs:
            self._dense = _emb_model.encode(self.docs, show_progress_bar=False, normalize_embeddings=True)

        # BM25（可選）
        if BM25Okapi is not None and self.docs:
            self._bm25 = BM25Okapi([_tokenize_for_bm25(d) for d in self.docs])

    def search(self, query: str, top_k_dense=10, top_k_bm25=10):
        scores = {}

        # Dense 相似度
        if self._dense is not None:
            q = _emb_model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
            # 餘弦相似度
            import numpy as np
            dense_scores = np.dot(self._dense, q)
            # 取 top_k_dense
            idxs = dense_scores.argsort()[-top_k_dense:][::-1]
            ds = dense_scores[idxs]
            # normalize 到 0..1
            if len(ds) > 1:
                ds_norm = (ds - ds.min()) / (ds.max() - ds.min() + 1e-9)
            else:
                ds_norm = ds
            for i, s in zip(idxs, ds_norm):
                scores[i] = max(scores.get(i, 0.0), float(s))

        # BM25 分數
        if self._bm25 is not None:
            bm25 = self._bm25.get_scores(_tokenize_for_bm25(query))
            import numpy as np
            idxs = bm25.argsort()[-top_k_bm25:][::-1]
            bs = bm25[idxs]
            if len(bs) > 1:
                bs_norm = (bs - bs.min()) / (bs.max() - bs.min() + 1e-9)
            else:
                bs_norm = bs
            for i, s in zip(idxs, bs_norm):
                # 取最大（dense / bm25）作為 hybrid 簡單融合
                scores[i] = max(scores.get(i, 0.0), float(s))

        # 如果兩者都沒有，就用關鍵詞命中數
        if not scores:
            q_terms = set(query.lower().split())
            for i, d in enumerate(self.docs):
                hit = sum(1 for t in q_terms if t in d.lower())
                if hit:
                    scores[i] = float(hit)

        # 按分數排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        hits = [{"text": self.docs[i], "meta": self.metas[i], "score": s} for i, s in ranked]
        return hits

def rerank_hits(query: str, hits: list[dict], top_k: int = 8):
    """
    可選重排：若有 CrossEncoder 就用；否則直接取前 top_k。
    """
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
    """
    把命中片段轉為 LLM 可吃的帶編號引文區塊。
    """
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

def filter_chunks_by_terms(chunks: list[dict], must_terms: list[str]) -> list[dict]:
    """在切塊後立即先過濾一次（降低後續噪音）。"""
    if not must_terms:
        return chunks
    out = []
    for c in chunks:
        if contains_any(c.get("text", ""), must_terms) or contains_any(c.get("title",""), must_terms):
            out.append(c)
    return out

def filter_hits_by_terms(hits: list[dict], must_terms: list[str]) -> list[dict]:
    """在檢索與重排後再過濾一次，確保最後進 LLM 的片段都含主題詞。"""
    if not must_terms:
        return hits
    out = []
    for h in hits:
        if contains_any(h.get("text",""), must_terms) or contains_any(h.get("meta",{}).get("title",""), must_terms):
            out.append(h)
    return out
 
# ========== Final Answer Synthesizer ==========
def synthesize_answer(query, graph_context, papers_context, model):
    llm = Ollama(model=model, temperature=0, base_url="http://localhost:11434")
    prompt = PromptTemplate(
        input_variables=["question", "graph", "papers"],
        template="""根據以下兩種來源，生成一個完整且專業的醫學回答：
        
        來源1 (GraphRAG 知識圖譜):
        {graph}

        來源2 (PapersRAG 最新醫學文獻):
        {papers}

        問題：
        {question}
        
        請僅引用 來源2 提供的 PMID 與段落，不要自行生成或捏造參考文獻。
        請整合以上資料，生成專業醫學回答。
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
    """
    Use LLM to break down the user's query into sub-queries and decide which agent should handle each.
    """
    llm = Ollama(model=llm_model, temperature=0, base_url="http://localhost:11434")
    parser = PydanticOutputParser(pydantic_object=QueryPlan)
    prompt = PromptTemplate(
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
你是一個醫學多代理系統的任務管理代理，負責將複雜的醫學問題拆解為子問題，並指派給不同的專家代理：

- GraphRAG Agent: 適合查詢結構化生物醫學知識圖譜。
- PapersRAG Agent: 適合查詢最新醫學文獻。

對以下問題進行分析，產生一個 JSON 格式的任務計畫，格式必須為：
{format_instructions}

問題：
{question}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(question=query)
    return parser.parse(output)

# ========== Main Logic ==========
if submit and query:
    with st.spinner("正在進行多代理協作查詢..."):
        try:
            # Step 1. Create task plan
            plan = create_query_plan(query, managing_model)
            st.info("🤖 Managing Agent 建立的查詢計畫：")
            st.json(plan.dict())

            graph_context = ""
            papers_context = ""

            # Step 2. Execute subtasks
            for task in plan.tasks:
                st.write(f"🔹 處理子問題: **{task.subquery}** → 指派給 **{task.agent}**")

                if task.agent == "GraphRAG":
                    results = retriever.search(query_text=task.subquery)
                    if results:
                        parsed_records = []
                        for cypher_query, result_data in results:
                            if isinstance(result_data, list):
                                for record in result_data:
                                    content = getattr(record, "content", None)
                                    if not content:
                                        continue
                                    match = re.search(r"properties=\{(.+?)\}", content)
                                    if match:
                                        props = ast.literal_eval(f"{{{match.group(1)}}}")
                                        parsed_records.append(props)
                                    else:
                                        parsed_records.append({"result": content})
                        if parsed_records:
                            graph_context += f"\n### 子問題: {task.subquery}\n" + "\n".join([str(r) for r in parsed_records])
                            st.subheader("📊 GraphRAG 子查詢結果")
                            st.dataframe(pd.DataFrame(parsed_records))
                        else:
                            st.warning(f"GraphRAG 對子問題 '{task.subquery}' 未找到結果")

                elif task.agent == "PapersRAG":
                    # A) 由子問題動態產生 must terms（只用於過濾片段，不重組 PubMed 查詢）
                    must_terms = derive_core_terms(task.subquery)

                    # B) 直接用原始子問題丟給 PubMed
                    raw_query = task.subquery
                    papers = retrieve_papers(raw_query, max_results=30)

                    if not papers:
                        st.warning(f"PapersRAG 對子問題 '{task.subquery}' 未找到相關論文")
                    else:
                        # C) 切塊 → 先做一次主題詞過濾（可保留也可拿掉，建議保留以避免離題）
                        all_chunks = []
                        for p in papers:
                            all_chunks.extend(build_chunks_from_entry(p))

                        filtered_chunks = filter_chunks_by_terms(all_chunks, must_terms)

                        if not filtered_chunks:
                            st.subheader("📄 PapersRAG 子查詢結果")
                            st.info("檢索到論文，但其內文片段沒有包含主題關鍵詞，僅列出論文以供人工檢視：")
                            for p in papers[:10]:
                                st.markdown(f"**{p.get('title','(no title)')}** ([link]({p['pubmed_url']}))")
                            papers_context += (
                                f"\n### 子問題: {task.subquery}\n"
                                + "\n".join([json.dumps(p, ensure_ascii=False) for p in papers[:10]])
                                + "\n\n(注意：此子問題在候選片段中未找到包含主題詞的內容。)"
                            )
                        else:
                            # D) 檢索與重排
                            index = PapersMiniIndex(filtered_chunks)
                            hits = index.search(task.subquery, top_k_dense=12, top_k_bm25=12)
                            hits = rerank_hits(task.subquery, hits, top_k=8)

                            # E) 最終再過濾一次（保守做法，確保送進 LLM 的片段都含主題詞）
                            hits = filter_hits_by_terms(hits, must_terms)

                            if not hits:
                                st.subheader("📄 PapersRAG 子查詢結果")
                                st.info("已過濾離題片段，但最終沒有符合主題詞的段落。以下為檢索到的論文清單：")
                                for p in papers[:10]:
                                    st.markdown(f"**{p.get('title','(no title)')}** ([link]({p['pubmed_url']}))")
                                papers_context += (
                                    f"\n### 子問題: {task.subquery}\n"
                                    + "\n".join([json.dumps(p, ensure_ascii=False) for p in papers[:10]])
                                    + "\n\n(注意：RAG 片段經主題詞過濾後為空。)"
                                )
                            else:
                                # F) 組裝帶編號引文 context
                                cited_context = make_cited_context(hits)

                                # UI 顯示
                                st.subheader("📄 PapersRAG 子查詢結果（RAG）")
                                for i, h in enumerate(hits, 1):
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

                                # G) 交給最終生成器使用
                                papers_context += f"\n### 子問題: {task.subquery}\n" + cited_context

            # Step 3. Final synthesis
            final_answer = synthesize_answer(query, graph_context, papers_context, answer_model)
            st.subheader("💬 最終回答")
            st.success(final_answer)

        except Exception as e:
            st.error(f"❌ 發生錯誤：{e}") # 轉 cypher 滿容易轉錯的，要是轉錯就會直接停掉，要改一下邏輯
