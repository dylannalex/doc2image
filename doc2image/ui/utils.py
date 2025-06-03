import streamlit as st


def rerun_with_commit(session):
    session.commit()
    st.rerun()
