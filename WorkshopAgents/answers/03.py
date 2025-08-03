def search_book(data: SearchBook) -> str:
    """
    Search the assets/moby_dict.txt file for a specific query.
    """
    from rank_bm25 import BM25Okapi

    def prep_docs() -> tuple[BM25Okapi, list[str]]:
        txt_path = "assets/moby_dick.txt"
        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            return None, []

        words = content.split()
        corpus = []
        chunk_size_words = 800
        overlap = int(chunk_size_words * 0.2)
        for i in range(0, len(words), chunk_size_words - overlap):
            corpus.append(" ".join(words[i : i + chunk_size_words]))

        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, corpus

    bm25, corpus = prep_docs()
    if bm25 is None or not corpus:
        return "Error: assets/moby_dict.txt not found or empty."

    query = data.query
    total_results = data.total_results
    tokenized_query = query.split(" ")
    results = bm25.get_top_n(tokenized_query, corpus, n=total_results)
    results_as_bullets = "\n".join(
        f"<result_{i + 1}>\n...{result}...\n</result_{i + 1}>"
        for i, result in enumerate(results)
    )

    return f"""
    <search_book>
    <search_query>{query}</search_query>
    {results_as_bullets}
    </search_book>
    """.strip()