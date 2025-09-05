import openai
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import torch
from src.classifier import HallucinationDetector
from dotenv import load_dotenv
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()


def detect_hallucination(context, question, reference_answer, classifier_path):
    # Generate answer using OpenAI
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    generated_answer = response['choices'][0]['message']['content']

    # Embed triplet
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    triplet = f"{context} [SEP] {question} [SEP] {generated_answer}"
    embedding = embedder.encode(triplet)

    # Load classifier
    model = HallucinationDetector()
    torch.save(model.state_dict(), "models/classifier.pt")
    model.load_state_dict(torch.load(classifier_path))
    model.eval()

    with torch.no_grad():
        score = model(torch.tensor(embedding).float().unsqueeze(0)).item()

    return {
        "generated_answer": generated_answer,
        "hallucination_score": score
    }
