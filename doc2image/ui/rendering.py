import streamlit as st
import pandas as pd

from doc2image.database import database_session_decorator
from doc2image import api


@database_session_decorator
def render_output(session, summary_session_id: int):
    summary_session = api.get_summary_by_id(session, summary_session_id)

    # -- Summary Session Details --
    st.markdown("#### 📄 Session Details")
    st.markdown(f"**Document:** {summary_session.document.name}")
    st.markdown(
        f"**Date:** {summary_session.generation_date.strftime('%Y-%m-%d %H:%M')}"
    )
    st.markdown(f"**LLM Model:** {summary_session.llm_model.name}")

    with st.expander("⚙️ Advanced Settings (View Only)"):
        st.markdown("**Document Summary LLM Settings**")
        st.code(
            f"Temperature: {summary_session.llm_temperature}\n"
            f"Top-p: {summary_session.llm_top_p}\n"
            f"Top-k: {summary_session.llm_top_k}\n"
        )
        st.markdown("**Prompt Generation LLM Settings**")
        st.code(
            f"Temperature: {summary_session.image_prompt_sessions[0].llm_temperature}\n"
            f"Top-p: {summary_session.image_prompt_sessions[0].llm_top_p}\n"
            f"Top-k: {summary_session.image_prompt_sessions[0].llm_top_k}\n"
        )
        st.markdown("**Chunking & Summary Settings**")
        st.code(
            f"Max Chunk Summary Size: {summary_session.max_chunk_summary_size}\n"
            f"Max Document Summary Size: {summary_session.max_document_summary_size}\n"
            f"Chunk Count: {len(summary_session.chunk_summaries)}"
        )

    # -- Generated Prompts
    st.markdown("#### 🖼️ Generated Prompts")

    prompts = []
    for prompt_session in summary_session.image_prompt_sessions:
        for prompt in prompt_session.prompts:
            prompts.append(prompt.prompt)
    if prompts:
        df_prompts = pd.DataFrame({"Prompt": prompts})
        st.dataframe(df_prompts, use_container_width=True)
        st.download_button(
            "Download Prompts as TXT",
            "\n\n".join(prompts),
            file_name="prompts.txt",
            mime="text/plain",
        )
    else:
        st.info("No prompts generated for this document.")

    # -- Generated Summary
    st.markdown("#### 📝 Generated Summary")
    st.text_area(
        "Summary",
        summary_session.document_summary,
        height=200,
    )
    st.download_button(
        "Download Summary as TXT",
        summary_session.document_summary,
        file_name="summary.txt",
        mime="text/plain",
    )

    # -- Chunk Summaries
    st.markdown("#### 📝 Chunk Summaries")
    chunk_data = [
        {
            "Chunk": i + 1,
            "Summary": chunk.chunk_summary,
        }
        for i, chunk in enumerate(summary_session.chunk_summaries)
    ]
    if chunk_data:
        st.dataframe(pd.DataFrame(chunk_data), use_container_width=True)
    else:
        st.info("No chunk summaries available.")
