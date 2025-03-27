def run():
    import streamlit as st
    import os
    from PIL import Image
    import numpy as np
    import pickle
    import tensorflow
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.layers import GlobalMaxPooling2D
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from sklearn.neighbors import NearestNeighbors
    from numpy.linalg import norm

    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model.trainable = False
    model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

    st.title("👕 Fashion Recommender")

    def save_uploaded_file(uploaded_file):
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    def feature_extraction(img_path, model):
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        expand_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img_array)
        result = model.predict(preprocessed_img).flatten()
        return result / norm(result)

    def recommend(features, feature_list):
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices

    uploaded_file = st.file_uploader("Upload your clothing item", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img_path = save_uploaded_file(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Item", width=250)

        features = feature_extraction(img_path, model)
        indices = recommend(features, feature_list)

        st.subheader("🔗 You May Also Like")
        cols = st.columns(5)

        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]], use_container_width=True)
                if st.button(f"Try On {i+1}", key=f"tryon_{i}"):
                    if "tryon_items" not in st.session_state:
                        st.session_state.tryon_items = []
                    st.session_state.tryon_items.append(filenames[indices[0][i]])
                    st.experimental_set_query_params(page="Try-On")
                    st.rerun()
