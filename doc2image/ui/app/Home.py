import os

import streamlit as st

st.set_page_config(page_title="Doc2Image", layout="wide", page_icon="🖼️")

st.title("🖼️ Doc2Image")

st.image("assets/robot-painting-a-landscape.png")

st.markdown(
    """
Turn your documents into creative prompts for AI image generators in seconds.


## How it works
1. **Upload** a PDF, TXT, or DOCX.
2. **Create** amazing image ideas.
3. **Generate** images using your favorite AI model (DALL-E, Midjourney, etc.).

## Features
- Fast, automatic document analysis
- Smart summaries and creative prompt generation
- Easy access to your prompt history

## Get started
Go to the **Generate Images** page to upload your document and start creating prompts.
"""
)
