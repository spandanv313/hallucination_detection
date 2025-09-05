import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.langchain_pipeline import detect_hallucination
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
st.title("Hallucination Detector")

context = st.text_area("Context")
question = st.text_input("Question")
reference_answer = st.text_input("Reference Answer")

if st.button("Detect"):
    result = detect_hallucination(
        context=context,
        question=question,
        reference_answer=reference_answer,
        classifier_path="models/classifier.pt"
    )
    st.write("Generated Answer:", result["generated_answer"])
    st.write("Hallucination Score:", result["hallucination_score"])
