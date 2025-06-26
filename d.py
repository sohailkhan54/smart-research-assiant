import streamlit as st
from pdfminer.high_level import extract_text
from transformers import pipeline
from io import BytesIO

def extract_pdf_text(uploaded_file):
    return extract_text(BytesIO(uploaded_file.read()))

def main():
    st.title("📚 Smart Research Assistant")

    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_pdf_text(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

        st.success("Document processed successfully!")

        summarizer = pipeline("summarization")
        qa_pipeline = pipeline("question-answering")
        generator = pipeline("text-generation", model="gpt2")

        st.subheader("📄 Document Summary")
        summary = summarizer(text[:1000], max_length=150, min_length=50, do_sample=False)
        st.write(summary[0]['summary_text'])

        mode = st.radio("Choose a mode:", ["Ask Anything", "Challenge Me"])

        if mode == "Ask Anything":
            user_question = st.text_input("Ask a question based on the document:")
            if user_question:
                answer = qa_pipeline(question=user_question, context=text)
                st.markdown(f"**Answer:** {answer['answer']}")
                st.markdown(f"_Justification (excerpt):_ `{answer['answer']}` was found in the context of the uploaded document.")

        elif mode == "Challenge Me":
            questions_output = ""
            if st.button("Generate Questions"):
                prompt = f"Create 3 logic-based comprehension questions from this document:\n{text[:1000]}"
                try:
                    questions_output = generator(prompt, max_length=300)[0]['generated_text']
                except:
                    questions_output = "Unable to generate questions. Please try again."
                st.text_area("Challenge Questions", value=questions_output, height=200)

            user_answer = st.text_input("Your answer to Q1:")
            correct_answer = questions_output.split("?")[0].split(":")[-1].strip() if questions_output else ""
            if user_answer:
                if user_answer.lower().strip() == correct_answer.lower():
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. Correct Answer: {correct_answer}")

if __name__ == "__main__":
    main()
