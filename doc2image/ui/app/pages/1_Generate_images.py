import os
import tempfile

import hydra
from hydra.core.global_hydra import GlobalHydra
import streamlit as st

from doc2image.ui.rendering import render_output
from doc2image import api


# --- Hydra Config Initialization ---
if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../../../configs", version_base=None)
cfg = hydra.compose(config_name="config")

# --- Streamlit Page Rendering ---
st.set_page_config(page_title="Doc2Image", layout="wide", page_icon="🖼️")

st.title("📝 Convert Document to Image")


def render_prompt_creation():
    st.markdown("### Upload Document")
    available_doc_formats = api.get_available_doc_formats()
    uploaded_file = st.file_uploader("Drop a document", type=available_doc_formats)

    total_prompts = st.number_input(
        "Total Prompts to Generate",
        value=cfg.pipeline.image_prompts_generator.total_prompts_to_generate,
        min_value=1,
    )

    st.markdown("### Generation Settings")

    col1, col2 = st.columns(2)
    with col1:
        # Provider selection
        available_providers = api.get_llm_providers()
        provider = st.selectbox("Provider", options=available_providers, index=0)
        st.session_state["provider"] = provider

        # API key input
        api_key = api.get_provider_api_key(provider)
        if provider == "OpenAI":
            api_key_input = st.text_input(
                "OpenAI API Key", type="password", value=api_key or ""
            )
            if api_key_input and api_key_input != api_key:
                api.update_provider_api_key(provider, api_key_input)
                st.session_state["openai_api_key"] = api_key_input
                st.rerun()
            else:
                st.session_state["openai_api_key"] = api_key
        else:
            st.session_state["openai_api_key"] = None

    with col2:
        # Filter models by provider
        llm_models = [
            m.model_name
            for m in api.get_all_llm_models()
            if m.provider_name == provider
        ]
        if llm_models:
            model_selected = st.selectbox("Select LLM model", options=llm_models)
            st.session_state["model_selected"] = model_selected
        else:
            st.warning(f"No {provider} models available. Please load one.")
            st.session_state["model_selected"] = None

        # Load new model
        model_name = st.text_input(f"Load New {provider} Model")
        if st.button("Load Model"):
            try:
                with st.spinner(f"Loading '{model_name}' model..."):
                    api.add_llm_model(
                        model_name=model_name,
                        provider_name=provider,
                        api_key=st.session_state.get("openai_api_key"),
                    )
                st.rerun()
            except Exception as e:
                st.toast(f"⚠️ {str(e)[:1000]}")

    with st.expander("⚙️ Advanced Configuration"):
        # Parser settings
        st.markdown("**Parser**")
        chunk_size = st.number_input("chunk_size", value=cfg.parser.chunk_size)
        chunk_overlap = st.number_input("chunk_overlap", value=cfg.parser.chunk_overlap)

        # Document Summarizer settings
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

        # Image Prompts Generator settings
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

    if uploaded_file and st.session_state.get("model_selected"):
        if st.button("🚀 Generate Images"):
            file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Processing document and generating prompts..."):
                # Prepare arguments for the API
                summarizer_args = {
                    "document_path": file_path,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "separators": cfg.parser.separators,
                    "is_separator_regex": cfg.parser.is_separator_regex,
                    "keep_separator": cfg.parser.keep_separator,
                    "strip_whitespace": cfg.parser.strip_whitespace,
                    "llm_api_key": st.session_state.get("openai_api_key"),
                    "llm_model_name": st.session_state["model_selected"],
                    "llm_temperature": doc_temp,
                    "llm_top_p": doc_top_p,
                    "llm_top_k": doc_top_k,
                    "llm_provider": provider,
                    "max_document_summary_size": max_document_summary_size,
                    "max_chunk_summary_size": max_chunk_summary_size,
                    "summarize_chunk_prompt_messages": cfg.prompts.summarize_chunk.messages,
                    "summarize_chunk_prompt_parameters": cfg.prompts.summarize_chunk.parameters,
                    "generate_document_summary_prompt_messages": cfg.prompts.generate_document_summary.messages,
                    "generate_document_summary_prompt_parameters": cfg.prompts.generate_document_summary.parameters,
                }
                summary_session_dto = api.summerize_document(**summarizer_args)

                prompts_generator_args = {
                    "summary_session_id": summary_session_dto.id,
                    "document_summary": summary_session_dto.document_summary,
                    "total_prompts_to_generate": total_prompts,
                    "generate_image_prompts_prompt_messages": cfg.prompts.generate_image_prompts.messages,
                    "generate_image_prompts_prompt_parameters": cfg.prompts.generate_image_prompts.parameters,
                    "llm_api_key": st.session_state.get("openai_api_key"),
                    "llm_model_name": st.session_state["model_selected"],
                    "llm_temperature": prompt_temp,
                    "llm_top_p": prompt_top_p,
                    "llm_top_k": prompt_top_k,
                    "provider_name": provider,
                }
                api.generate_image_prompts(**prompts_generator_args)

                st.session_state.generated_summary_id = summary_session_dto.id

            st.success("Pipeline completed! See results below or in History.")
            st.rerun()


def show_results():
    render_output(st.session_state.generated_summary_id)
    if st.button("Generate New"):
        st.session_state.generated_summary_id = None
        st.rerun()


if st.session_state.get("generated_summary_id") is None:
    render_prompt_creation()
else:
    show_results()
