import os
import tempfile

import streamlit as st
import hydra
from hydra.core.global_hydra import GlobalHydra

from doc2image.database import database_session_decorator
from doc2image.ui.rendering import render_output
from doc2image.ui.utils import rerun_with_commit
from doc2image import api


# --- Hydra Config Initialization ---
if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../../../configs", version_base=None)
cfg = hydra.compose(config_name="config")

# --- Streamlit Page Rendering ---
st.set_page_config(page_title="Doc2Image", layout="wide", page_icon="🖼️")

st.title("📝 Convert Document to Image")


@database_session_decorator
def render_prompt_creation(session):
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader("Drop a document", type=["pdf", "txt", "docx"])

    total_prompts = st.number_input(
        "Total Prompts to Generate",
        value=cfg.pipeline.image_prompts_generator.total_prompts_to_generate,
        min_value=1,
    )

    st.markdown("### Generation Settings")

    col1, col2 = st.columns(2)
    with col1:
        # Provider selection (defaults to OpenAI)
        provider_str = st.selectbox(
            "Provider", options=["OpenAI", "Ollama"], index=0, key="provider_select"
        )
        provider = provider_str.lower()
        st.session_state["provider"] = provider
        llm_models = [
            m.name
            for m in api.get_all_llm_models(session)
            if m.provider.name == provider
        ]

        # API key input (only for openai)
        api_key = api.get_provider_api_key(session, provider)
        if provider == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key_input",
                value=api_key,
            )
            if api_key:
                st.session_state["openai_api_key"] = api_key
                if st.button("Save API Key"):
                    api.update_provider_api_key(session, provider, api_key)
                    rerun_with_commit(session)

        else:
            st.session_state["openai_api_key"] = None
    with col2:
        # Filter models by provider
        if llm_models:
            model_selected = st.selectbox(
                "Select LLM model", options=llm_models, key="model_select"
            )
            st.session_state["model_selected"] = model_selected

        # Load new model (for both providers)
        model_name = st.text_input(
            f"Load New {provider_str} Model", key="model_name_input"
        )
        if st.button("Load Model"):
            try:
                api.add_llm_model(
                    session,
                    model_name=model_name,
                    provider_name=provider,
                    api_key=api_key,
                )
                rerun_with_commit(session)
            except Exception as e:
                error_msg = str(e)[0:1000]
                st.toast(f"⚠️ {error_msg}")

    if not llm_models:
        st.warning(f"No {provider_str} models available. Please load one.")
        model_selected = None
        st.session_state["model_selected"] = None

    with st.expander("⚙️ Advanced Configuration"):
        st.markdown("**Parser**")
        chunk_size = st.number_input(
            "chunk_size", value=cfg.parser.chunk_size, key="chunk_size"
        )
        chunk_overlap = st.number_input(
            "chunk_overlap", value=cfg.parser.chunk_overlap, key="chunk_overlap"
        )

        st.markdown("**Document Summarizer**")
        max_chunk_summary_size = st.number_input(
            "max_chunk_summary_size",
            value=cfg.pipeline.document_summarizer.max_chunk_summary_size,
        )
        max_document_summary_size = st.number_input(
            "max_document_summary_size",
            value=cfg.pipeline.document_summarizer.max_document_summary_size,
        )
        doc_temp = st.slider(
            "temperature",
            0.0,
            1.0,
            value=cfg.pipeline.document_summarizer.llm_params.temperature,
        )
        doc_top_p = st.slider(
            "top_p", 0.0, 1.0, value=cfg.pipeline.document_summarizer.llm_params.top_p
        )
        doc_top_k = st.number_input(
            "top_k",
            min_value=0,
            value=cfg.pipeline.document_summarizer.llm_params.top_k,
        )

        st.markdown("**Image Prompts Generator**")
        prompt_temp = st.slider(
            "prompt_temperature",
            0.0,
            1.0,
            value=cfg.pipeline.image_prompts_generator.llm_params.temperature,
        )
        prompt_top_p = st.slider(
            "prompt_top_p",
            0.0,
            1.0,
            value=cfg.pipeline.image_prompts_generator.llm_params.top_p,
        )
        prompt_top_k = st.number_input(
            "prompt_top_k",
            min_value=0,
            value=cfg.pipeline.image_prompts_generator.llm_params.top_k,
        )

    config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_chunk_summary_size": max_chunk_summary_size,
        "max_document_summary_size": max_document_summary_size,
        "doc_temp": doc_temp,
        "doc_top_p": doc_top_p,
        "doc_top_k": doc_top_k,
        "prompt_temp": prompt_temp,
        "prompt_top_p": prompt_top_p,
        "prompt_top_k": prompt_top_k,
    }

    if uploaded_file and st.session_state.get("model_selected"):
        if st.button("🚀 Generate Images"):
            file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            with st.spinner("Processing document and generating prompts..."):
                run_pipeline(
                    file_path, st.session_state["model_selected"], total_prompts, config
                )
            st.success("Pipeline completed! See results in History.")


@database_session_decorator
def run_pipeline(
    session,
    file_path: str,
    model_selected: str,
    total_prompts: int,
    config: dict,
):
    provider = st.session_state.get("provider", "openai")
    api_key = (
        st.session_state.get("openai_api_key", None) if provider == "openai" else None
    )
    summary_session = api.summerize_document(
        session,
        document_path=file_path,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=cfg.parser.separators,
        is_separator_regex=cfg.parser.is_separator_regex,
        keep_separator=cfg.parser.keep_separator,
        strip_whitespace=cfg.parser.strip_whitespace,
        llm_api_key=api_key,
        llm_model_name=model_selected,
        llm_temperature=config["doc_temp"],
        llm_top_p=config["doc_top_p"],
        llm_top_k=config["doc_top_k"],
        llm_provider=provider,
        max_document_summary_size=config["max_document_summary_size"],
        max_chunk_summary_size=config["max_chunk_summary_size"],
        summarize_chunk_prompt_messages=cfg.prompts.summarize_chunk.messages,
        summarize_chunk_prompt_parameters=cfg.prompts.summarize_chunk.parameters,
        generate_document_summary_prompt_messages=cfg.prompts.generate_document_summary.messages,
        generate_document_summary_prompt_parameters=cfg.prompts.generate_document_summary.parameters,
    )

    api.generate_image_prompts(
        session,
        summary_session=summary_session,
        document_path=file_path,
        document_summary=summary_session.document_summary,
        total_prompts_to_generate=total_prompts,
        generate_image_prompts_prompt_messages=cfg.prompts.generate_image_prompts.messages,
        generate_image_prompts_prompt_parameters=cfg.prompts.generate_image_prompts.parameters,
        llm_api_key=api_key,
        llm_model_name=model_selected,
        llm_temperature=config["prompt_temp"],
        llm_top_p=config["prompt_top_p"],
        llm_top_k=config["prompt_top_k"],
        provider_name=provider,
    )
    st.session_state.generated_summary_id = summary_session.id
    session.commit()
    st.rerun()


def show_results():
    render_output(st.session_state.generated_summary_id)
    if st.button("Back"):
        st.session_state.generated_summary_id = None
        st.rerun()


if st.session_state.get("generated_summary_id", None) is None:
    render_prompt_creation()
else:
    show_results()
