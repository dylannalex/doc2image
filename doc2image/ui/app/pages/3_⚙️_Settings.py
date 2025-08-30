import streamlit as st
import pandas as pd
from doc2image import api, schemas
from typing import List

# ----------------------------------------------------------------
# --- Helper & Handler Functions ---------------------------------
# ----------------------------------------------------------------

def handle_save_api_key(api_key: str):
    """Saves the OpenAI API key and provides user feedback."""
    if api_key:
        api.update_provider_api_key("OpenAI", api_key)
        st.toast("API Key saved!", icon="✅")
        st.rerun()
    else:
        st.toast("Please enter a key.", icon="⚠️")

def handle_add_model(provider: str, model_name: str):
    """Validates input and handles the logic for adding a new model."""
    if not model_name:
        st.warning("Please enter a model name.", icon="⚠️")
        return

    # Specific validation for OpenAI provider
    if provider == "OpenAI" and not api.get_provider_api_key("OpenAI"):
        st.error("You must save an OpenAI API key before adding an OpenAI model.", icon="❌")
        return

    # Proceed with adding the model
    try:
        with st.spinner(f"Verifying and adding '{model_name}'..."):
            api.add_llm_model(
                model_name=model_name,
                provider_name=provider,
                api_key=api.get_provider_api_key(provider),
            )
        st.toast(f"Model '{model_name}' added!", icon="🎉")
        st.cache_data.clear()  # Clear cache to reflect the new model
        st.rerun()
    except Exception as e:
        st.error(f"Failed to add model: {str(e)}", icon="🔥")

# ----------------------------------------------------------------
# --- UI Rendering Functions -------------------------------------
# ----------------------------------------------------------------

def render_header():
    """Sets up the page configuration and displays the main header."""
    st.set_page_config(page_title="Settings", layout="wide", page_icon="⚙️")
    st.title("⚙️ Settings")
    st.markdown("Add, view, and manage the AI models available for generating image ideas.")

def render_api_key_popover():
    """Renders the popover UI for managing the OpenAI API key."""
    is_key_set = bool(api.get_provider_api_key("OpenAI"))
    label = "Manage API Key" if is_key_set else "⚠️ Set API Key"
    
    with st.popover(label, use_container_width=True):
        st.markdown("**OpenAI API Key**")
        saved_key = api.get_provider_api_key("OpenAI")
        new_key = st.text_input(
            "api_key_input",
            type="password",
            value=saved_key or "",
            placeholder="sk-...",
            label_visibility="collapsed"
        )
        if st.button("Save Key"):
            handle_save_api_key(new_key)

def render_add_model_form():
    """Renders the complete form for adding a new model, including provider selection."""
    st.subheader("Add a New Model")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        provider = st.selectbox(
            "Select Provider",
            options=api.get_llm_providers(),
            key="provider_select",
            label_visibility="collapsed",
        )
    with col2:
        if provider == "OpenAI":
            render_api_key_popover()

    if provider == "OpenAI" and not api.get_provider_api_key("OpenAI"):
        st.warning("An OpenAI API key is required. Click 'Set API Key' above to add one.", icon="🔑")

    model_name = st.text_input("Model Name", placeholder="e.g., gpt-4o, llama3")
    
    c, *_ = st.columns(4)
    with c:
        if st.button("Add Model", type="primary", use_container_width=True):
            handle_add_model(provider, model_name)

def render_configured_models_list():
    """Fetches and displays the list of currently configured models in a dataframe."""
    st.subheader("Configured Models")

    @st.cache_data
    def get_cached_models() -> List[schemas.LlmModelProviderDTO]:
        return api.get_all_llm_models()

    all_models = get_cached_models()

    if not all_models:
        st.info("No models configured yet. Add one above to get started.")
    else:
        model_data = [{"Provider": m.provider_name, "Model Name": m.model_name} for m in all_models]
        df = pd.DataFrame(model_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------
# --- Main Page Execution ----------------------------------------
# ----------------------------------------------------------------

def main():
    """Main function to render the settings page."""
    render_header()
    with st.container(border=True):
        render_add_model_form()
        render_configured_models_list()

if __name__ == "__main__":
    main()