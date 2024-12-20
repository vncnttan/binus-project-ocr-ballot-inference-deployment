import streamlit as st
import tempfile
import os
from utils.cropping import cropping
from utils.predict import predict

st.title("Image Cropping App")
st.write("Upload an image to detect and crop regions")

chosen_model = st.selectbox("Choose a model", ["ResNet50.h5", "VGG19.h5", "RandomForest.pkl", "SVM.pkl"])
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    st.subheader("Original Image")
    st.image(uploaded_file, use_container_width=True)
    
    # Get Cropped Images
    cropped_images = cropping(tmp_path)
    
    if cropped_images:
        # Predict
        st.subheader("Prediction")
        predictions = predict(cropped_images, chosen_model, chosen_model != "ResNet50.h5" and chosen_model != "VGG19.h5")
        print(predictions)

        cols = st.columns(3)

        for idx, img in enumerate(cropped_images):

            with cols[idx % 3]:
                header_col, img_col = st.columns([1,1])
        
                with header_col:
                    if idx % 3 == 0:
                        st.subheader(f"Paslon 0{int(idx/3 + 1)}")
                        st.write(f"Prediction: {predictions[idx]}{predictions[idx+1]}{predictions[idx+2]}")

                with img_col:
                    st.image(img, caption=f"Region {idx+1} - Predicted: {predictions[idx]}", use_container_width=True)
    
    os.unlink(tmp_path)