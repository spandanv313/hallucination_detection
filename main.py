from src.langchain_pipeline import detect_hallucination
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    context = "India's independence movement was led by Mahatma Gandhi."
    question = "Who led India's independence?"
    reference_answer = "Mahatma Gandhi"

    result = detect_hallucination(
        context=context,
        question=question,
        reference_answer=reference_answer,
        classifier_path="models/classifier.pt"
    )

    print("Generated Answer:", result["generated_answer"])
    print("Hallucination Score:", result["hallucination_score"])
