from datetime import timedelta

import hydra
import streamlit as st
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from st_aggrid import AgGrid, GridOptionsBuilder

from doc2image.database import database_session_decorator
from doc2image.ui.rendering import render_output
from doc2image import api

# --- Hydra Config Initialization ---
if not GlobalHydra.instance().is_initialized():
    hydra.initialize(config_path="../../../configs", version_base=None)
cfg = hydra.compose(config_name="config")


# --- Streamlit Page Rendering ---
st.set_page_config(page_title="Doc2Image", layout="wide", page_icon="🖼️")

st.title("📚 History")


@database_session_decorator
def render_history(session):
    all_summary_sessions = api.get_all_document_summary_sessions(session)
    if not all_summary_sessions:
        st.info("No processed documents yet.")
        return

    data = []
    for s in all_summary_sessions:
        prompt_time = str(
            timedelta(seconds=round(s.image_prompt_sessions[0].session_time))
        )
        summary_time = str(timedelta(seconds=round(s.session_time)))
        data.append(
            {
                "Document": s.document.name,
                "Date": s.generation_date.strftime("%Y-%m-%d %H:%M"),
                "Prompt time": prompt_time,
                "Summary time": summary_time,
                "LLM Model": s.llm_model.name,
                "Prompts generated": sum(
                    len(ps.prompts) for ps in s.image_prompt_sessions
                ),
                "ID": s.id,
            }
        )

    df = pd.DataFrame(data)

    if not df.empty:
        st.markdown("#### Document Sessions")

        # Build AgGrid config
        gb = GridOptionsBuilder.from_dataframe(df.drop(columns=["ID"]))
        gb.configure_selection("single", use_checkbox=True)
        gb.configure_pagination(
            paginationAutoPageSize=True,
        )
        gb.configure_default_column(resizable=True, sortable=True, filterable=True)
        gb.configure_column("Date", sort="desc")

        grid_options = gb.build()
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=300,
            width="100%",
            allow_unsafe_jscode=True,
            update_mode="SELECTION_CHANGED",
        )
        selected_rows = grid_response["selected_rows"]

        if selected_rows is not None:
            selected_id = int(selected_rows.iloc[0]["ID"])
            st.session_state.selected_summary_id = selected_id

        else:
            st.session_state.selected_summary_id = None

    else:
        st.info("No processed documents yet.")

    if st.session_state.get("selected_summary_id", None) is not None:
        render_output(st.session_state.selected_summary_id)


render_history()
