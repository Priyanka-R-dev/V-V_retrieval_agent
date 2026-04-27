import os
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Cache for LLM-generated query expansions to avoid redundant API calls
_expansion_cache = {}


def _get_llm():
    """Create and return the Azure OpenAI LLM instance."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
    )


def generate_query_expansions(question, llm):
    """Use LLM to generate alternative search queries for better retrieval."""
    if question in _expansion_cache:
        return _expansion_cache[question]

    try:
        messages = [
            SystemMessage(content=(
                "You are a search query expansion assistant. Given a question about a document, "
                "generate exactly 3 alternative search queries that might match the document text. "
                "Return ONLY the 3 queries, one per line, no numbering or bullets."
            )),
            HumanMessage(content=f"Question: {question}")
        ]
        response = llm.invoke(messages)
        expansions = [
            line.strip()
            for line in response.content.strip().split('\n')
            if line.strip() and len(line.strip()) > 5
        ][:3]
        _expansion_cache[question] = expansions
        return expansions
    except Exception:
        return []


def get_expanded_docs(question, retriever, llm):
    """Retrieve docs using original question + LLM-generated expanded queries.

    Returns list of (Document, score) tuples, re-ranked by hybrid scoring.
    """
    results = retriever.invoke(question)

    # Generate expansions via LLM
    expansions = generate_query_expansions(question, llm)
    seen_contents = {doc.page_content for doc, _ in results}

    for exp_query in expansions:
        extra_results = retriever.invoke(exp_query)
        for doc, score in extra_results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                results.append((doc, score))

    # Hybrid re-ranking: combine similarity score with keyword overlap
    question_lower = question.lower()
    keywords = [w for w in question_lower.split() if len(w) > 3]

    def hybrid_score(doc_score_tuple):
        doc, sim_score = doc_score_tuple
        content_lower = doc.page_content.lower()
        keyword_hits = sum(1 for k in keywords if k in content_lower)
        keyword_ratio = keyword_hits / max(len(keywords), 1)
        return 0.7 * sim_score + 0.3 * keyword_ratio

    results.sort(key=hybrid_score, reverse=True)

    return results


def generate_answer(question, retriever, max_retries=3):
    load_dotenv()
    llm = _get_llm()
    scored_docs = get_expanded_docs(question, retriever, llm)

    # Separate docs and scores
    docs = [doc for doc, _ in scored_docs]
    scores = [score for _, score in scored_docs]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )

    print(f"\n    📄 Retrieved {len(docs)} chunks (avg score: {avg_score:.3f}):")
    for i, (doc, score) in enumerate(scored_docs):
        preview = doc.page_content[:80].replace('\n', ' ')
        src = doc.metadata.get('source', '?')
        print(f"       [{i+1}] Score {score:.3f} | {src}: {preview}...")

    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    messages = [
        SystemMessage(content=(
            "You are a document analysis assistant for regulatory validation documents. "
            "Answer the question based ONLY on the provided context from the uploaded documents. "
            "IMPORTANT RULES:\n"
            "1. ☒ and [x] mean CHECKED/SELECTED. ☐ and [ ] mean UNCHECKED/NOT SELECTED.\n"
            "2. If the question asks 'what type' or 'which', list ALL items marked with ☒ or [x] — do NOT skip any.\n"
            "3. If the question is Yes/No, find the EXACT checkbox pair for THAT specific question. "
            "Do NOT use a checkbox from a neighboring question.\n"
            "4. Distinguish between 'plan' and 'summary report' — they are different questions.\n"
            "5. Read the FULL context carefully before answering. Do NOT stop at the first match.\n"
            "6. Be concise and specific.\n"
            "7. If the context contains ANY relevant information, use it to answer.\n"
            "8. ONLY respond with 'NA' if the context has absolutely zero information about the topic.\n"
            "9. When multiple checkboxes are marked ☒ or [x], include ALL of them in your answer.\n"
            "10. CRITICAL: The context may contain MULTIPLE checkbox questions close together. "
            "Each question has its OWN Yes/No checkboxes. Match each checkbox to the question text "
            "IMMEDIATELY above it. Do NOT mix up checkboxes from different questions.\n"
            "11. When asked 'who will execute/perform', look for specific team names (e.g., 'Team A') "
            "rather than generic role titles.\n"
            "12. For questions about engagement or participation, consider the OVERALL intent of the section, "
            "not just a single isolated checkbox.\n"
            "13. If a checkbox appears WITHOUT a clear question immediately before it (orphaned checkbox), "
            "do NOT use that checkbox as your answer. Instead, reason from the surrounding context.\n"
            "14. When the context says testing is done 'by representative users' from 'all user areas' "
            "and the Business Owner reviews/approves results, that means business units ARE engaged."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ]

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            answer = response.content

            # Flag low-confidence answers
            if avg_score < similarity_threshold:
                answer = f"[Low Confidence] {answer}"

            return answer, docs, scores
        except Exception as e:
            err_str = str(e)
            if "content_filter" in err_str or "ResponsibleAIPolicyViolation" in err_str:
                print(f"    ⚠️ Content filter triggered — retrying with sanitized context...")
                # Trim context to reduce chance of triggering filter
                trimmed_context = "\n\n".join(
                    f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content[:300]}"
                    for d in docs[:5]
                )
                messages[1] = HumanMessage(
                    content=f"Context:\n{trimmed_context}\n\nQuestion: {question}\n\nAnswer:"
                )
                try:
                    response = llm.invoke(messages)
                    answer = response.content
                    if avg_score < similarity_threshold:
                        answer = f"[Low Confidence] {answer}"
                    return answer, docs, scores
                except Exception:
                    return "[Content Filter] Unable to generate answer — Azure content policy triggered by document text.", docs, scores
            elif "429" in err_str or "rate_limit" in err_str:
                wait = 30 * (attempt + 1)
                print(f"    ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise e

    return "Rate limit exceeded. Try again later.", docs, scores