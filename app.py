import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NFC–ACO Lung Segmentation",
    page_icon="🫁",
    layout="wide"
)

MODEL_PATH = "models/Best_Final_UNetPlusPlus_NFC_ACO.keras"
IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_segmentation_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model


# try:
#     model = load_segmentation_model()
#     model_loaded = True
# except Exception as e:
#     model_loaded = False
#     model_error = str(e)

try:
    with st.spinner("Loading NFC–ACO U-Net++ model..."):
        model = load_segmentation_model()

    model_loaded = True
   

except Exception as e:
    model_loaded = False
    st.error("❌ Model loading failed")
    st.exception(e)
    st.stop()
# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

def preprocess_image(image):

    # Final model expects RGB input
    image = image.convert("RGB")

    # Resize to model input size
    resized = image.resize(IMAGE_SIZE)

    image_array = np.array(resized).astype(np.float32)

    # Same normalisation used during training
    if image_array.max() > 1.5:
        image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return resized, image_array


# --------------------------------------------------
# CREATE OVERLAY
# --------------------------------------------------

def create_overlay(original_image, mask):

    image = np.array(original_image.convert("RGB")).astype(np.float32)

    # Resize mask back to original image size
    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    mask_image = mask_image.resize(
        original_image.size,
        Image.Resampling.NEAREST
    )

    mask_array = np.array(mask_image) / 255.0

    overlay = image.copy()

    # Highlight predicted lung region
    overlay[:, :, 1] = np.where(
        mask_array > 0.5,
        np.minimum(
            overlay[:, :, 1] * 0.6 + 100,
            255
        ),
        overlay[:, :, 1]
    )

    return overlay.astype(np.uint8)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🫁 NFC–ACO-Inspired Lung Segmentation")

st.write(
    """
    Research prototype demonstrating lung segmentation from chest
    X-ray images using the U-Net++ configuration selected through
    NFC–ACO-inspired hyperparameter optimisation.
    """
)

st.warning(
    "Research prototype only — not intended for clinical diagnosis "
    "or medical decision-making."
)


# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

if model_loaded:

    st.success("✓ NFC–ACO U-Net++ model loaded successfully")

else:

    st.error("Model could not be loaded.")

    with st.expander("Technical error"):
        st.code(model_error)

    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.divider()

st.header("Chest X-ray Lung Segmentation")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    resized_image, model_input = preprocess_image(
        original_image
    )

    with st.spinner("Segmenting lung region..."):

        prediction = model.predict(
            model_input,
            verbose=0
        )

    # Remove batch dimension
    probability_mask = prediction[0]

    # Remove final channel if output is H x W x 1
    if probability_mask.ndim == 3:
        probability_mask = probability_mask[:, :, 0]

    # Binary segmentation
    binary_mask = (
        probability_mask >= THRESHOLD
    ).astype(np.uint8)

    overlay = create_overlay(
        original_image,
        binary_mask
    )


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Original X-ray")

        st.image(
            original_image,
            use_container_width=True
        )


    with col2:

        st.subheader("Predicted Lung Mask")

        st.image(
            binary_mask * 255,
            clamp=True,
            use_container_width=True
        )


    with col3:

        st.subheader("Segmentation Overlay")

        st.image(
            overlay,
            use_container_width=True
        )


    # --------------------------------------------------
    # PREDICTION INFORMATION
    # --------------------------------------------------

    st.subheader("Segmentation Information")

    predicted_percentage = (
        binary_mask.mean() * 100
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Predicted Mask Coverage",
        f"{predicted_percentage:.2f}%"
    )

    c2.metric(
        "Segmentation Threshold",
        f"{THRESHOLD:.2f}"
    )


# --------------------------------------------------
# RESEARCH RESULTS
# --------------------------------------------------

st.divider()

st.header("Dissertation Experimental Results")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "12-candidate NFC–ACO",
    "0.520755"
)

c2.metric(
    "12-candidate Random",
    "0.487162"
)

c3.metric(
    "Final Test Dice",
    "0.95612"
)

c4.metric(
    "Final Test IoU",
    "0.91593"
)

st.caption(
    "Proxy optimisation scores and final segmentation scores "
    "represent different evaluation stages and should not be "
    "directly compared."
)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.header("Selected Hyperparameters")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Learning Rate", "0.00005")
c2.metric("Batch Size", "4")
c3.metric("Dropout", "0.1")
c4.metric("Optimizer", "RMSprop")


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

st.divider()

st.header("About the Research")

st.write(
    """
    This dissertation investigates an NFC–ACO-inspired
    hyperparameter optimisation strategy for lung segmentation.

    NFC-inspired exploration is combined with ACO-style
    reinforcement to guide candidate selection. The proposed
    strategy was compared with Random Search under equal
    candidate-evaluation budgets.
    """
)