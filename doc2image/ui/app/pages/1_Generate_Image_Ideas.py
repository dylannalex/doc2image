import os
import tempfile
from random import choice

import streamlit as st
import hydra
from hydra.core.global_hydra import GlobalHydra

from doc2image.ui.rendering import render_output
from doc2image import api, schemas

# ----------------------------------------------------------------
# --- Hydra Config Initialization --------------------------------
# ----------------------------------------------------------------

if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../../../configs", version_base=None)
cfg = hydra.compose(config_name="config")

# ----------------------------------------------------------------
# --- Helper & Handler Functions ---------------------------------
# ----------------------------------------------------------------


def _get_random_generation_message():
    return choice(
        (
            "The creative spark is igniting... Please wait. ✨",
            "Our AI is sketching out some fresh concepts... 🎨",
            "Analyzing your document and dreaming up ideas... 🧠",
            "Firing up the inspiration engine... 🚀",
            "Your document is being transformed into image ideas...✨",
            "Our friendly robot is hard at work on your request... 🧠",
            "Unlocking the visual potential of your document... 💡",
            "The AI is now imagining visuals for you... Please wait. 🚀",
        )
    )


def handle_generation(
    uploaded_file, model_name, provider, total_ideas, doc_temp, prompt_temp
):
    """
    Handles the end-to-end process of generating image ideas from a document,
    displaying a simple spinner during the process.
    """
    api_key = api.get_provider_api_key(provider)

    # Use st.spinner for a clean, single-message loading indicator.
    with st.spinner(_get_random_generation_message(), show_time=True):
        try:
            # Step 1: Reading and processing your document
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f"_{uploaded_file.name}",
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            # Step 2: Summarizing the document's content
            summarizer_args = build_summarizer_args(
                file_path, uploaded_file.name, model_name, provider, api_key, doc_temp
            )
            summary_session_dto = api.summerize_document(**summarizer_args)

            # Step 3: Crafting creative image ideas
            prompts_generator_args = build_prompts_generator_args(
                summary_session_dto,
                model_name,
                provider,
                api_key,
                total_ideas,
                prompt_temp,
            )
            api.generate_image_prompts(**prompts_generator_args)

            # On success, update the session state
            st.session_state.generated_summary_id = summary_session_dto.id

            # Show a success message before rerunning to the results page
            st.success("Ideas generated successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during generation: {str(e)}")
        finally:
            # Ensure the temporary file is always cleaned up
            if "file_path" in locals() and os.path.exists(file_path):
                os.remove(file_path)


def build_summarizer_args(
    file_path, original_filename, model_name, provider, api_key, temperature
):
    """Builds the dictionary of arguments for the summarization API call."""
    return {
        "document_path": file_path,
        "file_name": original_filename,
        "llm_model_name": model_name,
        "llm_provider": provider,
        "llm_api_key": api_key,
        "llm_temperature": temperature,
        "chunk_size": cfg.parser.chunk_size,
        "chunk_overlap": cfg.parser.chunk_overlap,
        "separators": cfg.parser.separators,
        "is_separator_regex": cfg.parser.is_separator_regex,
        "keep_separator": cfg.parser.keep_separator,
        "strip_whitespace": cfg.parser.strip_whitespace,
        "llm_top_p": cfg.pipeline.document_summarizer.llm_params.top_p,
        "llm_top_k": cfg.pipeline.document_summarizer.llm_params.top_k,
        "max_document_summary_size": cfg.pipeline.document_summarizer.max_document_summary_size,
        "max_chunk_summary_size": cfg.pipeline.document_summarizer.max_chunk_summary_size,
        "summarize_chunk_prompt_messages": cfg.prompts.summarize_chunk.messages,
        "summarize_chunk_prompt_parameters": cfg.prompts.summarize_chunk.parameters,
        "generate_document_summary_prompt_messages": cfg.prompts.generate_document_summary.messages,
        "generate_document_summary_prompt_parameters": cfg.prompts.generate_document_summary.parameters,
    }


def build_prompts_generator_args(
    summary_dto, model_name, provider, api_key, total_prompts, temperature
):
    """Builds the dictionary of arguments for the image prompt generation API call."""
    return {
        "summary_session_id": summary_dto.id,
        "document_summary": summary_dto.document_summary,
        "total_prompts_to_generate": total_prompts,
        "llm_model_name": model_name,
        "provider_name": provider,
        "llm_api_key": api_key,
        "llm_temperature": temperature,
        "llm_top_p": cfg.pipeline.image_prompts_generator.llm_params.top_p,
        "llm_top_k": cfg.pipeline.image_prompts_generator.llm_params.top_k,
        "generate_image_prompts_prompt_messages": cfg.prompts.generate_image_prompts.messages,
        "generate_image_prompts_prompt_parameters": cfg.prompts.generate_image_prompts.parameters,
    }


# ----------------------------------------------------------------
# --- UI Rendering Functions -------------------------------------
# ----------------------------------------------------------------


def render_header():
    st.set_page_config(page_title="Generate Ideas", layout="wide", page_icon="✨")
    st.title("✨ Generate Image Ideas")
    st.markdown("Bring your documents to life by turning them into a list of creative, AI-generated image ideas.")


def render_generation_form():
    """Renders the main form, grouping all inputs for a single submission."""

    with st.form(key="generation_form"):
        # --- Step 1, 2, 3 ---
        st.subheader("1. Upload Your Document")
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=api.get_available_doc_formats(),
            label_visibility="collapsed",
        )
        st.write("")

        st.subheader("2. Choose Your Creative Engine")

        @st.cache_data
        def get_cached_models() -> list[schemas.LlmModelProviderDTO]:
            return api.get_all_llm_models()

        all_models = get_cached_models()
        if not all_models:
            st.warning(
                "No models found. Please add a model in the settings page first.",
                icon="⚠️",
            )
            st.page_link("pages/3_⚙️_Settings.py", label="Go to Settings", icon="⚙️")
            st.form_submit_button("🚀 Generate Ideas", disabled=True)

            return
        
        model_options = {
            f"{m.model_name} ({m.provider_name})": (m.model_name, m.provider_name)
            for m in all_models
        }
        selected_model_display = st.selectbox(
            "Select a model", options=model_options.keys(), label_visibility="collapsed"
        )
        st.write("")

        st.subheader("3. Customize Your Request")
        total_ideas = st.number_input(
            "Number of Image Ideas to Generate", min_value=1, max_value=20, value=10
        )

        # --- Fine-Tune Expander ---
        with st.expander("Fine-Tune Request"):
            col1, col2 = st.columns(2)
            with col1:
                doc_temp = st.slider(
                    label="Summary Creativity",
                    key="doc_temp",  # Use key to identify the widget
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help=(
                        "Controls how the AI interprets your document.\n\n"
                        "- **Low values (e.g., 0.2):** The AI will be very literal and stick strictly to the text. Good for technical documents.\n\n"
                        "- **High values (e.g., 0.9):** The AI will be more interpretive and creative in finding connections. Good for artistic or abstract source material."
                    ),
                )
            with col2:
                prompt_temp = st.slider(
                    label="Idea Originality",
                    key="prompt_temp",  # Use key to identify the widget
                    min_value=0.0,
                    max_value=1.0,
                    value=0.85,
                    help=(
                        "Controls the 'imagination' of the final image ideas.\n\n"
                        "- **Low values (e.g., 0.3):** Generates predictable, straightforward ideas directly from the summary.\n\n"
                        "- **High values (e.g., 1.0):** Encourages wilder, more artistic, and unexpected concepts."
                    ),
                )

        st.write("")
        # --- Form Submission Button ---
        submitted = st.form_submit_button(
            "🚀 Generate Ideas", type="primary", width="stretch"
        )

    if submitted:
        if not uploaded_file:
            st.warning("Please upload a document before generating ideas.", icon="📄")
        else:
            model_name, provider = model_options[selected_model_display]
            handle_generation(
                uploaded_file, model_name, provider, total_ideas, doc_temp, prompt_temp
            )


def render_results_view():
    """Renders the results view and a button to start a new generation."""
    render_output(st.session_state.generated_summary_id)
    if st.button("✨ Generate New Ideas", width="stretch", type="primary"):
        st.session_state.generated_summary_id = None
        st.rerun()


# ----------------------------------------------------------------
# --- Main Page Execution ----------------------------------------
# ----------------------------------------------------------------


def main():
    render_header()
    if "generated_summary_id" not in st.session_state:
        st.session_state.generated_summary_id = None

    if st.session_state.generated_summary_id:
        render_results_view()
    else:
        render_generation_form()


if __name__ == "__main__":
    main()
