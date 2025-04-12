import json
import os
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
from whoosh.query import Every

from difflib import SequenceMatcher

def string_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def build_merged_index(json_file_paths, index_dir="product_index"):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    schema = Schema(
        asin=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        description=TEXT(stored=True),
        features=TEXT(stored=True),
        main_category=TEXT(stored=True),
        store=TEXT(stored=True)
    )

    ix = create_in(index_dir, schema)
    writer = ix.writer()

    for json_file in json_file_paths:
        with open(json_file, 'r') as f:
            products_dict = json.load(f)

        for asin, product in products_dict.items():
            title = str(product.get("title", ""))
            description = " ".join(product.get("description", [])) if isinstance(product.get("description", []), list) else str(product.get("description", ""))
            features = " ".join(product.get("features", [])) if isinstance(product.get("features", []), list) else str(product.get("features", ""))

            category = str(product.get("main_category", ""))
            store = str(product.get("store", ""))

            writer.add_document(
                asin=asin,
                title=title,
                description=description,
                features=features,
                main_category=category,
                store=store
            )

    writer.commit()

def search_index(query_str, index_dir="product_index", top_k=5):
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "description", "features", "main_category", "store"], schema=ix.schema)
        query = parser.parse(query_str)
        results = searcher.search(query, limit=top_k)

        for hit in results:
            print(f"ASIN: {hit['asin']}")
            print(f"Title: {hit['title']}")
            print(f"Store: {hit['store']}")
            print(f"Category: {hit['main_category']}")
            print(f"Description: {hit['description']}\n")
            
    return results

def search_index_return_json(query_str, index_dir="product_index", top_k=100):
    ix = open_dir(index_dir)
    results_list = []

    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "description", "features", "main_category", "store"], schema=ix.schema)
        query = parser.parse(query_str)
        results = searcher.search(query, limit=top_k)

        for hit in results:
            result = {
                "asin": hit["asin"],
                "title": hit["title"],
                "description": hit["description"],
                "features": hit.get("features", ""),
                "main_category": hit.get("main_category", ""),
                "store": hit.get("store", "")
            }
            results_list.append(result)

    return results_list

def search_index_best_effort(query_str, index_dir="product_index", top_k=100):
    ix = open_dir(index_dir)
    results_list = []

    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "description", "features", "main_category", "store"], schema=ix.schema)
        query = parser.parse(query_str)

        # First attempt: normal query
        results = searcher.search(query, limit=top_k)

        if len(results) == 0:
            # print("No direct matches found. Returning top results overall.")
            # Fallback: match everything, return best-scored results manually
            results = searcher.search(Every(), limit=top_k)

        for hit in results:
            result = {
                "asin": hit["asin"],
                "title": hit["title"],
                "description": hit["description"],
                "features": hit.get("features", ""),
                "main_category": hit.get("main_category", ""),
                "store": hit.get("store", "")
            }
            results_list.append(result)

    return results_list

def search_index_best_effort_v2(query_str, index_dir="product_index", top_k=10, fuzzy=True):
    ix = open_dir(index_dir)
    results_list = []
    fallback = 0


    with ix.searcher() as searcher:
        fields = ["title", "description", "features", "main_category", "store"]
        parser = MultifieldParser(fields, schema=ix.schema)

        # Use fuzzy matching (~2) if enabled
        if fuzzy:
            fuzzy_query_str = " OR ".join([f'{field}:"{query_str}"~2' for field in fields])
            query = parser.parse(fuzzy_query_str)
        else:
            query = parser.parse(query_str)

        results = searcher.search(query, limit=top_k)

        # Fallback if nothing found
        if len(results) == 0:
            print("No direct matches found. Finding nearest matches...")
            all_docs = searcher.search(Every(), limit=None)
            scored = []

            for hit in all_docs:
                title = hit.get("title", "")
                sim = string_similarity(query_str, title)
                scored.append((sim, hit))

            # Sort by similarity and pick top_k
            scored.sort(reverse=True, key=lambda x: x[0])
            results = [hit for _, hit in scored[:top_k]]

        for hit in results:
            result = {
                "asin": hit["asin"],
                "title": hit["title"],
                "description": hit["description"],
                "features": hit.get("features", ""),
                "main_category": hit.get("main_category", ""),
                "store": hit.get("store", "")
            }
            results_list.append(result)
    return results_list, fallback