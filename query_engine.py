import json
import os
from dotenv import load_dotenv
from retriever import load_retriever
from question_loader import load_questions_from_output
from answer_generator import generate_answer

def run_query_engine():
    load_dotenv()
    output_file = os.getenv("OUTPUT_FILE")
    if not output_file:
        raise ValueError("OUTPUT_FILE env var must be set (path to questionnaire PDF/txt/json).")
    results_file = os.getenv("RESULTS_FILE", "validation_answers.json")
    results_md = os.getenv("RESULTS_MD", "validation_answers.md")

    # Derive title from the output filename
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    doc_title = base_name.replace('_', ' ').replace('-', ' ')

    print(f"Extracting questions from: {output_file}")
    questions = load_questions_from_output(output_file)
    print(f"Found {len(questions)} questions.\n")

    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q}")

    print(f"\nLoading retriever...")
    retriever = load_retriever(k=int(os.getenv("TOP_K", "10")))

    results = []
    md_output = [f"# {doc_title} - Generated Answers\n"]

    for idx, question in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] {question[:70]}...")
        answer, sources, scores = generate_answer(question, retriever)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        is_low_confidence = answer.startswith("[Low Confidence]")

        results.append({
            "question": question,
            "answer": answer,
            "avg_score": round(avg_score, 3),
            "confidence": "low" if is_low_confidence else "normal",
            "sources": [
                {
                    "source": doc.metadata.get("source", "N/A"),
                    "score": round(scores[i], 3) if i < len(scores) else None,
                }
                for i, doc in enumerate(sources)
            ]
        })

        md_output.append(f"## Q{idx}: {question}")
        md_output.append(f"**Answer:** {answer}\n")

    # Save JSON
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save Markdown
    with open(results_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_output))

    print(f"\n✅ Answers saved to:")
    print(f"   JSON: {results_file}")
    print(f"   Markdown: {results_md}")

if __name__ == "__main__":
    run_query_engine()