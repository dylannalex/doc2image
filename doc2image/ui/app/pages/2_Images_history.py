from datetime import timedelta

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

from doc2image.ui.rendering import render_output
from doc2image import api

# --- Streamlit Page Rendering ---
st.set_page_config(page_title="Idea Gallery", layout="wide", page_icon="🖼️")

st.title("🖼️ Your Idea Gallery")
st.markdown("Step back into your gallery of past creations, where every document tells a visual story.")

def render_history():
    all_summary_sessions = api.get_all_document_summary_sessions()
    if not all_summary_sessions:
        st.info("No processed documents yet.")
        return

    data = []
    for summary_session in all_summary_sessions:
        prompt_time = "N/A"
        if summary_session.image_prompt_sessions:
            prompt_time = str(
                timedelta(
                    seconds=round(summary_session.image_prompt_sessions[0].session_time)
                )
            )

        summary_time = str(timedelta(seconds=round(summary_session.session_time)))
        data.append(
            {
                "Document": summary_session.document.name,
                "Date": summary_session.generation_date.strftime("%Y-%m-%d %H:%M"),
                "Prompt time": prompt_time,
                "Summary time": summary_time,
                "LLM Model": summary_session.llm_model.name,
                "Prompts generated": sum(
                    len(ps.prompts) for ps in summary_session.image_prompt_sessions
                ),
                "ID": summary_session.id,
            }
        )

    df = pd.DataFrame(data)

    if not df.empty:
        st.markdown("#### Document Sessions")
        gb = GridOptionsBuilder.from_dataframe(df.drop(columns=["ID"]))
        gb.configure_selection("single", use_checkbox=True)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(resizable=True, sortable=True, filterable=True)
        gb.configure_column("Date", sort="desc")

        grid_response = AgGrid(
            df,
            gridOptions=gb.build(),
            height=300,
            width="100%",
            allow_unsafe_jscode=True,
            update_on=["selectionChanged"]
        )
        selected_rows = grid_response["selected_rows"]

        if selected_rows is not None:
            selected_id = int(selected_rows.iloc[0]["ID"])
            st.session_state.selected_summary_id = selected_id

        else:
            st.session_state.selected_summary_id = None

    else:
        st.info("No processed documents yet.")

    if st.session_state.get("selected_summary_id") is not None:
        render_output(st.session_state.selected_summary_id)


render_history()
