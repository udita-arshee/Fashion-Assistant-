import streamlit as st
from streamlit_option_menu import option_menu
import main
import tryon_main

st.set_page_config(page_title="Fashion Assistant", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Recommendation", "Try-On"],
        icons=["search", "person"],
        menu_icon="shirt",
        default_index=0,
    )

if selected == "Recommendation":
    main.run()
elif selected == "Try-On":
    tryon_main.run()
