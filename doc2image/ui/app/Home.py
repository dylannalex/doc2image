import streamlit as st

st.set_page_config(page_title="Doc2Image", layout="wide", page_icon="🖼️")

st.title("🖼️ Doc2Image")

st.image("assets/robot-painting-a-landscape.png")

st.markdown(
    """
Doc2Image is an AI-powered app that transforms your documents into creative image ideas. Just upload a file (PDF, TXT, DOCX, Markdown and more) and Doc2Image will read the content, highlight the key points, and create visual descriptions ready to use with your favorite image generation platforms like MidJourney, DALL·E, ChatGPT, and more.

### 📚 How It Works

1. **Upload a document** — PDF, DOCX, Markdown, TXT, and more are supported.
2. **Choose a model** — OpenAI or a local Ollama model  
3. **Customize your request** — Select how many image prompts you want, and (optionally) tweak advanced settings like temperature or chunk size  
4. **Generate amazing image ideas** — Doc2Image transforms your document into stunning, ready-to-use prompts

### 🚀 Get started

Go to the **Generate Images** page to upload your document and start creating amazing images.

### ❤️ Support the project

If you enjoy using this project, please consider [giving it a star ⭐️ on GitHub](https://github.com/dylannalex/doc2image) — it helps others discover it too!

"""
)
