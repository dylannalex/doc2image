import streamlit as st

st.set_page_config(page_title="Doc2Image", layout="wide", page_icon="🎨")

st.title(
    "🖼️ Welcome to Doc2Image",
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Transform Your Documents into Stunning Visual Ideas")
    st.markdown(
        """
        **Doc2Image is here to help you unlock your creativity!** Upload any document (`PDF`, `DOCX`, `TXT` and more), and let the AI generate a list of unique image ideas, ready for you to use. Perfect for blog posts, presentations, or simply sparking your imagination!
        """
    )

    st.info(
        "**Heads-Up:** This app creates image ideas, not the final images. You can then use these ideas in any AI image generator you like!",
        icon="💡",
    )

    st.subheader("✨ Key Features")
    st.markdown(
        """
        - **Intuitive Interface:** A clean, guided experience from start to finish.
        - **Flexible AI:** Works with both OpenAI models and local models via Ollama.
        - **Idea History:** Never lose a great idea with the built-in idea gallery.
        - **Customizable:** Fine-tune the AI's creativity to get the perfect results.
        """
    )
    st.subheader("🚀 How to Get Started")

    st.page_link(
        "pages/3_⚙️_Settings.py",
        label="**1\\. Configure Your Models (One-Time Setup)**",
        # icon="⚙️",
    )
    st.markdown("Add your API keys and choose which AI models you want to use.")

    st.page_link(
        "pages/1_Generate_Image_Ideas.py",
        label="**2\\. Generate Image Ideas**",
        # icon="🚀",
    )

    st.markdown("Upload a document and let the magic happen!")

    st.page_link(
        "pages/2_Idea_Gallery.py",
        label="**3\\. Visit Your Idea Gallery**",
        # icon="🖼️",
    )
    st.markdown("Browse your creations and use them in your favorite image generator.")

with col2:
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.image("assets/robot-painting-a-landscape.png", width="stretch")

# --- Footer ---
st.markdown(
    """
    ---
    <div style="text-align: center;">
        If you enjoy this project, please consider giving it a star ⭐️ on <a href="https://github.com/dylannalex/doc2image" target="_blank">GitHub</a> to help others discover it!
    </div>
    """,
    unsafe_allow_html=True,
)
