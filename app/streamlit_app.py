"""Minimal serving shell; model loading is added after the core protocol is locked."""

import streamlit as st

st.set_page_config(page_title="TrustRec", layout="wide")
st.title("TrustRec")
st.caption("Evidence-aware recommendation demo shell")
st.info(
    "The recommendation pipeline is not fitted yet. Use the protocol and task backlog "
    "to build the first snapshot."
)
