import streamlit as st
from transformers import pipeline
st.set_page_config(page_title="Text Summarizer", layout="centered")



@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()


def split_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]



st.title(" Text Summarizer Chatbot")

st.markdown("""


-  Upload a `.txt` file, or  
-  Paste your text below  
-  Click **Summarize** to get the summary
""")


uploaded_file = st.file_uploader("Upload a text file", type=["txt"])


user_input = st.text_area("Or paste your text here:", height=200)


if st.button(" Summarize"):
    if uploaded_file:
        long_text = uploaded_file.read().decode("utf-8")
    elif user_input.strip():
        long_text = user_input
    else:
        st.warning(" upload a file or paste text.")
        st.stop()

    
    chunks = split_text(long_text)
    summaries = []

    with st.spinner(" please wait"):
        for chunk in chunks:
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])

    final_summary = "\n".join(summaries)


    st.subheader(" Final Summary")
    st.text_area("Summary", final_summary, height=300)

    st.download_button("Download Summary", final_summary, file_name="summary.txt")
