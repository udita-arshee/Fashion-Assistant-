import streamlit as st
from PIL import Image
import os

def run():
    st.title("👕 Multi-Try-On Fashion Preview")

    base = Image.open("base_model.png").convert("RGBA")

    uploaded_images = st.file_uploader("Upload clothing items", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Allow session-passed items from recommender
    if "tryon_items" in st.session_state:
        for file_path in st.session_state.tryon_items:
            if os.path.exists(file_path):
                uploaded_images.append(open(file_path, "rb"))
        st.session_state.tryon_items = []  # clear after loading

    if uploaded_images:
        tryon_preview = base.copy()

        for idx, uploaded_file in enumerate(uploaded_images):
            img = Image.open(uploaded_file).convert("RGBA")
            st.sidebar.subheader(f"🪄 Item {idx+1}")

            scale = st.sidebar.slider(f"Scale Item {idx+1} (%)", 10, 300, 100)
            move_x = st.sidebar.slider(f"Move Left ↔ Right (Item {idx+1})", -568, 568, 0)
            move_y = st.sidebar.slider(f"Move Up ↕ Down (Item {idx+1})", -1600, 1600, int(base.height / 2))

            img_resized = img.resize((
                int(img.width * scale / 100),
                int(img.height * scale / 100)
            ))

            tryon_preview.paste(img_resized, box=(move_x, move_y), mask=img_resized)

        st.subheader("🧵 Try-On Preview")
        st.image(tryon_preview, width=400)
