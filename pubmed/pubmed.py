"""
PubMed Full-Text Retrieval and Embedding Pipeline

This script performs:
1. PubMed search using Entrez
2. Check for PMC full-text availability
3. Download full-text from Europe PMC
4. Store structured data in JSON
5. Embed full-text using a BioNLP model
6. Store embeddings in ChromaDB
"""

from Bio import Entrez
import requests
import json
from bs4 import BeautifulSoup
import os
# from sentence_transformers import SentenceTransformer
# import chromadb
from tqdm import tqdm

Entrez.email = "zzxcth717@gmail.com"  # Replace with your email

def search_pubmed(keyword, max_results=30):
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=max_results, sort = "relevance")
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

# 新增漂亮格式的 JSON 儲存函數
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# def embed_texts(texts, model_name="allenai/specter"):
#     model = SentenceTransformer(model_name)
#     return model.encode(texts)

# def store_in_chromadb(docs, embeddings, metadatas, collection_name="pubmed_fulltext"):
#     client = chromadb.Client()
#     collection = client.get_or_create_collection(collection_name)
#     collection.add(
#         documents=docs,
#         embeddings=embeddings,
#         metadatas=metadatas,
#         ids=[f"doc_{i}" for i in range(len(docs))]
#     )

def main():
    while True:
        keyword = input("Enter a keyword for PubMed search: ").strip()
        if not keyword or keyword.lower() == "exit":
            print("Exiting.")
            break
        max_results = 20
        pmids = search_pubmed(keyword, max_results=max_results)

        print(pmids)

        entries = []
        fulltext_entries = []

        for pmid in tqdm(pmids, desc="Saving links"):
            pmc_id = get_pmc_id(pmid)
            title = get_title_from_pubmed(pmid)

            entry = {
                "pmid": pmid,
                "title": title,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }

            if pmc_id:
                entry["pmc_id"] = pmc_id
                entry["pmc_url"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
                xml = get_fulltext_from_europe_pmc(pmc_id)
                if xml:
                    structured_text = extract_fulltext_structure(xml)
                    entry["fulltext"] = structured_text
                    fulltext_entries.append(entry)
                else:
                    entry["fulltext"] = None
            else:
                entry["fulltext"] = None

            entries.append(entry)

        # 儲存漂亮格式的 JSON
        os.makedirs("result", exist_ok=True)
        output_filename = os.path.join("result", f"{keyword.replace(' ', '_')}.json")
        valid_entries = [e for e in entries if e["fulltext"]]
        if not valid_entries:
            print("No fulltext found. Skipping save.")
        else:
            save_to_json(valid_entries, filename=output_filename)

if __name__ == "__main__":
    main()
