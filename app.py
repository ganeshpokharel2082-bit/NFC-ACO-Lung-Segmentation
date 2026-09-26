import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import tensorflow as tf
from pathlib import Path
import time


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NFC–ACO Research Workflow",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

MODEL_PATH = "models/Best_Final_UNetPlusPlus_NFC_ACO.keras"

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

SAMPLE_DIR = Path("demo_samples")
MASK_DIR = SAMPLE_DIR / "mask"
HPO_CSV = Path("FINAL_72_HPO_Evaluations.csv")


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2.25rem;
    }

    .flow {
        display: flex;
        gap: .45rem;
        align-items: center;
        flex-wrap: wrap;
        margin: .8rem 0 1.2rem 0;
    }

    .step {
        padding: .55rem .75rem;
        border: 1px solid rgba(128,128,128,.32);
        border-radius: 12px;
        font-weight: 600;
    }

    .arrow {
        opacity: .55;
        font-size: 1.15rem;
    }

    .research-card {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 15px;
        height: 100%;
    }

    .best {
        padding: 1rem 1.2rem;
        border: 2px solid #2e7d32;
        border-radius: 14px;
        margin-top: 12px;
    }

    .small {
        opacity: .75;
        font-size: .92rem;
    }

    .stage-box {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 14px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


try:

    model = load_model()
    model_loaded = True

except Exception as e:

    model = None
    model_loaded = False
    model_error = str(e)


# ============================================================
# 5. PREPROCESS X-RAY
# ============================================================

def preprocess_image(image):

    # Final model expects RGB input
    rgb = image.convert("RGB")

    # Resize to training/model input size
    resized = rgb.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    arr = np.array(
        resized
    ).astype(np.float32)

    # Normalisation used during training
    if arr.max() > 1.5:
        arr = arr / 255.0

    # Add batch dimension
    model_input = np.expand_dims(
        arr,
        axis=0
    )

    return resized, model_input


# ============================================================
# 6. MODEL PREDICTION
# ============================================================

def predict_mask(model, image):

    resized, model_input = preprocess_image(image)

    prediction = model.predict(
        model_input,
        verbose=0
    )[0]

    # Remove channel dimension if H x W x 1
    if prediction.ndim == 3:
        prediction = prediction[:, :, 0]

    binary_mask = (
        prediction >= THRESHOLD
    ).astype(np.uint8)

    return resized, prediction, binary_mask

def overlay_image(image, mask):
    """
    Overlay the predicted binary lung mask on the chest X-ray.
    Green areas represent the predicted lung region.
    """

    # Convert image to RGB and resize to model dimensions
    base = np.array(
        image.convert("RGB").resize(IMAGE_SIZE)
    ).astype(np.float32)

    # Make sure mask is 2D
    mask = np.squeeze(mask)

    # Convert mask to boolean
    mask_bool = mask > 0.5

    # Create overlay
    overlay = base.copy()

    # Green transparent-looking overlay
    overlay[:, :, 1] = np.where(
        mask_bool,
        np.minimum(overlay[:, :, 1] * 0.55 + 110, 255),
        overlay[:, :, 1]
    )

    # Slightly reduce red and blue inside mask
    overlay[:, :, 0] = np.where(
        mask_bool,
        overlay[:, :, 0] * 0.75,
        overlay[:, :, 0]
    )

    overlay[:, :, 2] = np.where(
        mask_bool,
        overlay[:, :, 2] * 0.75,
        overlay[:, :, 2]
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)
# ============================================================
# 7. CREATE SEGMENTATION OVERLAY
# ============================================================

def create_overlay(image, mask):

    base = np.array(
        image.convert("RGB").resize(
            IMAGE_SIZE
        )
    ).astype(np.float32)

    overlay = base.copy()

    # Green highlight for predicted lung region
    overlay[:, :, 1] = np.where(
        mask > 0.5,
        np.minimum(
            overlay[:, :, 1] * 0.55 + 110,
            255
        ),
        overlay[:, :, 1]
    )

    return overlay.astype(np.uint8)




def get_demo_pairs():

    pairs = []

    if not SAMPLE_DIR.exists():
        return pairs

    if not MASK_DIR.exists():
        return pairs

    for xray_path in sorted(
        SAMPLE_DIR.glob("*.png")
    ):

        mask_path = MASK_DIR / xray_path.name

        if mask_path.exists():

            pairs.append(
                {
                    "id": xray_path.stem,
                    "xray": xray_path,
                    "mask": mask_path
                }
            )

    return pairs


samples = get_demo_pairs()


# ============================================================
# 8B. LOAD RECORDED HPO EXPERIMENTS
# ============================================================

@st.cache_data
def load_hpo_results():
    if not HPO_CSV.exists():
        return None

    df = pd.read_csv(HPO_CSV)

    required = {
        "seed", "evaluation_number", "learning_rate", "batch_size",
        "dropout", "optimizer", "best_val_dice", "best_val_iou",
        "epochs_completed", "candidate_time_seconds", "method"
    }

    if not required.issubset(df.columns):
        return None

    df = df.copy()
    df["evaluation_number"] = pd.to_numeric(df["evaluation_number"], errors="coerce")
    df["best_val_dice"] = pd.to_numeric(df["best_val_dice"], errors="coerce")
    df["best_val_iou"] = pd.to_numeric(df["best_val_iou"], errors="coerce")
    df["candidate_time_seconds"] = pd.to_numeric(df["candidate_time_seconds"], errors="coerce")
    df = df.dropna(subset=["evaluation_number", "best_val_dice", "method", "seed"])
    df["evaluation_number"] = df["evaluation_number"].astype(int)
    df["seed"] = df["seed"].astype(int)
    return df.sort_values(["seed", "method", "evaluation_number"]).reset_index(drop=True)


def prepare_seed_run(df, seed, method):
    run = df[(df["seed"] == int(seed)) & (df["method"] == method)].copy()
    run = run.sort_values("evaluation_number").reset_index(drop=True)
    run["running_best_dice"] = run["best_val_dice"].cummax()
    run["cumulative_time_seconds"] = run["candidate_time_seconds"].fillna(0).cumsum()
    return run


def format_lr(value):
    return f"{float(value):.0e}"


# ============================================================
# 9. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

    <h1>🫁 NFC–ACO-Inspired U-Net++ Lung Segmentation</h1>

    <div>
    Interactive dissertation demonstrator:
    from chest X-ray preprocessing to restricted-budget
    hyperparameter optimisation and final segmentation.
    </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.warning(
    "Research prototype only — not intended for clinical "
    "diagnosis or medical decision-making."
)


# ============================================================
# 10. RESEARCH WORKFLOW
# ============================================================

st.markdown(
    """
    <div class="flow">

    <div class="step">1 · Dataset</div>
    <div class="arrow">→</div>

    <div class="step">2 · Preprocessing</div>
    <div class="arrow">→</div>

    <div class="step">3 · U-Net++</div>
    <div class="arrow">→</div>

    <div class="step">4 · NFC Exploration</div>
    <div class="arrow">→</div>

    <div class="step">5 · Proxy Dice</div>
    <div class="arrow">→</div>

    <div class="step">6 · ACO Exploitation</div>
    <div class="arrow">→</div>

    <div class="step">7 · Best Configuration</div>
    <div class="arrow">→</div>

    <div class="step">8 · Final Evaluation</div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 11. MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🖼️ Image Pipeline",
        "🧬 NFC–ACO Search",
        "📊 Experimental Evidence",
        "ℹ️ Research Summary"
    ]
)


# ============================================================
# TAB 1 — IMAGE PIPELINE
# ============================================================

with tab1:

    st.header(
        "From Chest X-ray to Lung Mask"
    )

    st.caption(
        "Fixed examples from the dissertation dataset are used "
        "here; public image upload is intentionally disabled."
    )


    # --------------------------------------------------------
    # CHECK SAMPLES
    # --------------------------------------------------------

    if not samples:

        st.error(
            "No paired demonstration samples were found. "
            "Place X-rays directly inside demo_samples/ and "
            "matching masks inside demo_samples/mask/ using "
            "the same filename."
        )

    else:

        ids = [
            sample["id"]
            for sample in samples
        ]

        sid = st.selectbox(
            "Choose a fixed demonstration sample",
            ids,
            index=0,
            key="demo_sample_selector"
        )

        selected = next(
            sample
            for sample in samples
            if sample["id"] == sid
        )

        xray_path = selected["xray"]
        mask_path = selected["mask"]

        xray = Image.open(
            xray_path
        )

        ground_truth = Image.open(
            mask_path
        ).convert("L")


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        resized, model_input = preprocess_image(
            xray
        )

        gt256 = ground_truth.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        gt_array = np.array(
            gt256
        )

        # Dissertation ground-truth mask threshold
        gt_binary = (
            gt_array > 127
        ).astype(np.uint8)


        # ----------------------------------------------------
        # WORKFLOW STAGE SELECTOR
        # ----------------------------------------------------

        stage = st.radio(
            "Workflow stage",
            [
                "Original",
                "Preprocessed",
                "Ground Truth",
                "U-Net++ Prediction",
                "Overlay"
            ],
            horizontal=True,
            key="image_workflow_stage"
        )


        st.divider()


        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        if stage == "Original":

            c1, c2 = st.columns(
                [1.2, 1]
            )

            with c1:

                st.image(
                    xray,
                    caption=f"Sample {sid}: original chest X-ray",
                    width="stretch"
                )

            with c2:

                st.subheader(
                    "Stage 1 — Dataset"
                )

                st.write(
                    "The original chest radiograph is the "
                    "starting point of the segmentation pipeline."
                )

                st.metric(
                    "Demonstration sample",
                    sid
                )

                st.metric(
                    "Original dimensions",
                    f"{xray.size[0]} × {xray.size[1]}"
                )


        # ====================================================
        # PREPROCESSING
        # ====================================================

        elif stage == "Preprocessed":

            c1, c2 = st.columns(2)

            with c1:

                st.image(
                    xray,
                    caption="Original chest X-ray",
                    width="stretch"
                )

            with c2:

                st.image(
                    resized,
                    caption="Preprocessed model input: 256 × 256",
                    width="stretch"
                )

            st.info(
                "Preprocessing used by the deployed model: "
                "RGB conversion → resize to 256 × 256 → "
                "normalisation to [0,1]."
            )


        # ====================================================
        # GROUND TRUTH
        # ====================================================

        elif stage == "Ground Truth":

            c1, c2 = st.columns(2)

            with c1:

                st.image(
                    resized,
                    caption="Preprocessed chest X-ray",
                    width="stretch"
                )

            with c2:

                st.image(
                    gt_binary * 255,
                    caption="Ground-truth lung mask",
                    clamp=True,
                    width="stretch"
                )

            st.info(
                "The corresponding reference mask is resized "
                "using nearest-neighbour interpolation and "
                "binarised using the dissertation threshold."
            )


        # ====================================================
        # U-NET++ PREDICTION
        # ====================================================

        elif stage == "U-Net++ Prediction":

            if not model_loaded:

                st.error(
                    "The trained model could not be loaded."
                )

                st.code(
                    model_error
                )

            else:

                with st.spinner(
                    "Running U-Net++ segmentation..."
                ):

                    _, probability_mask, binary_mask = (
                        predict_mask(
                            model,
                            xray
                        )
                    )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.image(
                        resized,
                        caption="Model input",
                        width="stretch"
                    )

                with c2:

                    st.image(
                        probability_mask,
                        caption="U-Net++ probability map",
                        clamp=True,
                        width="stretch"
                    )

                with c3:

                    st.image(
                        binary_mask * 255,
                        caption="Predicted binary lung mask",
                        clamp=True,
                        width="stretch"
                    )

                st.caption(
                    "Prediction probabilities are converted to "
                    f"a binary segmentation using threshold "
                    f"{THRESHOLD:.2f}."
                )


        # ====================================================
        # OVERLAY
        # ====================================================

        elif stage == "Overlay":

            if not model_loaded:

                st.error(
                    "The trained model could not be loaded."
                )

                st.code(
                    model_error
                )

            else:

                with st.spinner(
                    "Generating final segmentation..."
                ):

                    _, probability_mask, binary_mask = (
                        predict_mask(
                            model,
                            xray
                        )
                    )

                    overlay = create_overlay(
                        resized,
                        binary_mask
                    )


                c1, c2, c3 = st.columns(3)

                with c1:

                    st.image(
                        resized,
                        caption="Chest X-ray",
                        width="stretch"
                    )

                with c2:

                    st.image(
                        binary_mask * 255,
                        caption="Predicted lung mask",
                        clamp=True,
                        width="stretch"
                    )

                with c3:

                    st.image(
                        overlay,
                        caption="Segmentation overlay",
                        width="stretch"
                    )


                predicted_percentage = (
                    binary_mask.mean() * 100
                )

                m1, m2 = st.columns(2)

                m1.metric(
                    "Predicted Mask Coverage",
                    f"{predicted_percentage:.2f}%"
                )

                m2.metric(
                    "Prediction Threshold",
                    f"{THRESHOLD:.2f}"
                )


        # ----------------------------------------------------
        # OPTIONAL AUTOMATED WALKTHROUGH
        # ----------------------------------------------------

        st.divider()

        if st.button(
    "▶ Run Image Pipeline Demonstration",
            key="run_image_pipeline",
            use_container_width=False
        ):
            import time

            st.markdown("---")
            st.subheader("🔬 Live Image Segmentation Workflow")

            progress = st.progress(0)
            status = st.empty()
            stage_box = st.empty()

            # --------------------------------------------------
            # STEP 1 — ORIGINAL IMAGE
            # --------------------------------------------------
            status.markdown("### Step 1 of 7 — Loading Chest X-ray")
            stage_box.info(
                "The original chest radiograph is selected from the dissertation dataset."
            )

            progress.progress(10)
            time.sleep(0.7)

            c1, c2 = st.columns([1, 1])

            with c1:
                st.image(
                    xray,
                    caption="Original chest X-ray",
                    width="stretch"
                )

            with c2:
                st.markdown("""
                **Input stage**

                The chest X-ray is the original image supplied to the
                preprocessing pipeline before segmentation.
                """)

            # --------------------------------------------------
            # STEP 2 — RESIZE
            # --------------------------------------------------
            status.markdown("### Step 2 of 7 — Resizing Image")
            stage_box.info(
                "The chest X-ray is resized to 256 × 256 pixels."
            )

            progress.progress(25)
            time.sleep(0.7)

            c1, c2 = st.columns(2)

            with c1:
                st.image(
                    xray,
                    caption="Original image",
                    width="stretch"
                )

            with c2:
                st.image(
                    resized,
                    caption="Resized input — 256 × 256",
                    width="stretch"
                )

            # --------------------------------------------------
            # STEP 3 — NORMALISATION
            # --------------------------------------------------
            status.markdown("### Step 3 of 7 — Pixel Normalisation")
            stage_box.info(
                "Pixel intensities are converted to floating-point values "
                "and normalised to the range [0, 1]."
            )

            progress.progress(40)
            time.sleep(0.7)

            st.markdown("""
            **Preprocessing**

            `Original image → RGB conversion → Resize 256 × 256 → Normalise [0,1]`
            """)

            # --------------------------------------------------
            # STEP 4 — GROUND TRUTH
            # --------------------------------------------------
            status.markdown("### Step 4 of 7 — Ground-Truth Lung Mask")
            stage_box.info(
                "The corresponding reference lung mask is displayed for comparison."
            )

            progress.progress(55)
            time.sleep(0.7)

            c1, c2 = st.columns(2)

            with c1:
                st.image(
                    resized,
                    caption="Preprocessed chest X-ray",
                    width="stretch"
                )

            with c2:
                st.image(
                    gt256,
                    caption="Ground-truth lung mask",
                    width="stretch"
                )

            # --------------------------------------------------
            # STEP 5 — U-NET++ INFERENCE
            # --------------------------------------------------
            status.markdown("### Step 5 of 7 — U-Net++ Segmentation")
            stage_box.info(
                "The preprocessed image is passed through the trained "
                "U-Net++ segmentation model."
            )

            progress.progress(70)
            time.sleep(0.7)

            if model_loaded:

                resized_pred, probability_map, predicted_mask = predict_mask(
                    model,
                    xray
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.image(
                        resized_pred,
                        caption="Model input",
                        width="stretch"
                    )

                with c2:
                    st.image(
                        probability_map,
                        caption="U-Net++ probability map",
                        clamp=True,
                        width="stretch"
                    )

                with c3:
                    st.image(
                        predicted_mask * 255,
                        caption="Binary prediction",
                        clamp=True,
                        width="stretch"
                    )

            else:
                st.error("The trained U-Net++ model could not be loaded.")
                st.stop()

            # --------------------------------------------------
            # STEP 6 — THRESHOLDING
            # --------------------------------------------------
            status.markdown("### Step 6 of 7 — Prediction Threshold")
            stage_box.info(
                f"A threshold of {THRESHOLD:.2f} converts the probability "
                "output into a binary lung segmentation mask."
            )

            progress.progress(85)
            time.sleep(0.7)

            coverage = float(np.mean(predicted_mask)) * 100

            m1, m2 = st.columns(2)

            m1.metric(
                "Prediction Threshold",
                f"{THRESHOLD:.2f}"
            )

            m2.metric(
                "Predicted Mask Coverage",
                f"{coverage:.2f}%"
            )

            # --------------------------------------------------
            # STEP 7 — FINAL OVERLAY
            # --------------------------------------------------
            status.markdown("### Step 7 of 7 — Segmentation Result")
            stage_box.success(
                "The predicted lung mask is overlaid on the chest X-ray "
                "to visualise the segmented region."
            )

            progress.progress(100)
            time.sleep(0.7)

            final_overlay = overlay_image(
                resized_pred,
                predicted_mask
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.image(
                    resized_pred,
                    caption="Chest X-ray",
                    width="stretch"
                )

            with c2:
                st.image(
                    predicted_mask * 255,
                    caption="Predicted lung mask",
                    clamp=True,
                    width="stretch"
                )

            with c3:
                st.image(
                    final_overlay,
                    caption="Segmentation overlay",
                    width="stretch"
                )

            st.success(
                "✓ Image segmentation workflow complete — "
                "the chest X-ray has passed through preprocessing, "
                "U-Net++ inference, thresholding and final mask visualisation."
            )
        # if st.button(
        #     "▶ Run Image Pipeline Demonstration",
        #     key="run_image_demo"
        # ):

        #     demo_placeholder = st.empty()

        #     stages = [
        #         "1. Dataset sample selected",
        #         "2. Resize to 256 × 256",
        #         "3. Normalise pixel values to [0,1]",
        #         "4. Load ground-truth mask",
        #         "5. Pass image through U-Net++",
        #         "6. Apply 0.50 segmentation threshold",
        #         "7. Generate binary lung mask",
        #         "8. Create segmentation overlay"
        #     ]

            # for i, description in enumerate(
            #     stages,
            #     start=1
            # ):

            #     demo_placeholder.info(
            #         f"Step {i}/8 — {description}"
            #     )

            #     time.sleep(0.45)

            # demo_placeholder.success(
            #     "✓ Image segmentation workflow complete."
            # )


# ============================================================
# TAB 2 — NFC–ACO SEARCH
# ============================================================

with tab2:

    st.header("NFC–ACO Recorded Experiment Replay")
    st.caption(
        "Replay of the recorded dissertation optimisation experiment. "
        "The browser does not retrain the candidate models."
    )

    hpo_df = load_hpo_results()

    # --------------------------------------------------------
    # SEARCH SPACE
    # --------------------------------------------------------
    st.subheader("Search Space and Primary Evaluation Protocol")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Discrete search space", "48 configs")
    m2.metric("Primary budget", "12 evaluations")
    m3.metric("Repeated seeds", "42 · 123 · 2026")
    m4.metric("Fitness", "Validation Dice")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**Learning rate**  \n`1e-5` · `5e-5` · `1e-4` · `5e-4`")
    c2.markdown("**Batch size**  \n`4` · `8`")
    c3.markdown("**Dropout**  \n`0.0` · `0.1` · `0.2`")
    c4.markdown("**Optimiser**  \n`Adam` · `RMSprop`")

    st.info(
        "Low-fidelity candidate evaluation: 180 training images → "
        "80 validation images → maximum 5 epochs → early stopping patience 2."
    )

    st.divider()

    if hpo_df is None:
        st.error(
            "The recorded HPO file could not be loaded. Place "
            "FINAL_72_HPO_Evaluations.csv in the same folder as app_v2.py."
        )

    else:
        available_seeds = sorted(hpo_df["seed"].unique().tolist())

        control1, control2 = st.columns([1, 1])
        with control1:
            replay_seed = st.selectbox(
                "Choose experiment seed",
                available_seeds,
                index=0,
                key="hpo_replay_seed"
            )
        with control2:
            replay_speed = st.select_slider(
                "Replay speed",
                options=["Fast", "Normal", "Slow"],
                value="Normal",
                key="hpo_replay_speed"
            )

        delay = {"Fast": 0.12, "Normal": 0.35, "Slow": 0.70}[replay_speed]

        nfc_run = prepare_seed_run(hpo_df, replay_seed, "NFC-ACO")
        random_run = prepare_seed_run(hpo_df, replay_seed, "Random Search")
        replay_budget = min(len(nfc_run), len(random_run), 12)

        st.markdown(
            "### What the replay shows\n"
            "Each step uses a **real recorded candidate** from the CSV. "
            "For NFC–ACO, a new running-best candidate is described as promising "
            "search information for ACO-style reinforcement. The replay does not "
            "invent or display pheromone values that were not recorded in the CSV."
        )

        if st.button(
            "▶ Replay Recorded 12-Candidate Experiment",
            key="replay_recorded_hpo",
            type="primary",
            use_container_width=True
        ):
            progress = st.progress(0)
            status_box = st.empty()
            candidate_box = st.empty()
            metric_box = st.empty()
            chart_box = st.empty()

            nfc_best = -np.inf
            random_best = -np.inf
            chart_rows = []

            for i in range(replay_budget):
                nfc = nfc_run.iloc[i]
                rnd = random_run.iloc[i]

                previous_nfc_best = nfc_best
                nfc_best = max(nfc_best, float(nfc["best_val_dice"]))
                random_best = max(random_best, float(rnd["best_val_dice"]))
                is_new_best = float(nfc["best_val_dice"]) > previous_nfc_best

                status_box.markdown(
                    f"### Evaluation {i + 1} of {replay_budget} — Seed {replay_seed}"
                )

                reinforcement_text = (
                    "**Promising candidate:** a new running-best proxy Dice was recorded. "
                    "Its hyperparameter choices provide stronger information for the "
                    "ACO-style exploitation stage."
                    if is_new_best else
                    "**Candidate evaluated:** it did not exceed the current running best, "
                    "so the existing best search information is retained."
                )

                candidate_box.markdown(
                    f"""
                    <div class="stage-box">
                    <b>NFC–ACO candidate {int(nfc['evaluation_number'])}</b><br>
                    Learning rate: <b>{format_lr(nfc['learning_rate'])}</b> ·
                    Batch: <b>{int(nfc['batch_size'])}</b> ·
                    Dropout: <b>{float(nfc['dropout']):.1f}</b> ·
                    Optimiser: <b>{str(nfc['optimizer']).upper()}</b><br><br>
                    Recorded validation Dice: <b>{float(nfc['best_val_dice']):.6f}</b> ·
                    IoU: <b>{float(nfc['best_val_iou']):.6f}</b> ·
                    Epochs: <b>{int(nfc['epochs_completed'])}</b> ·
                    Candidate time: <b>{float(nfc['candidate_time_seconds']):.1f} s</b><br><br>
                    {reinforcement_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with metric_box.container():
                    q1, q2, q3, q4 = st.columns(4)
                    q1.metric("NFC–ACO current Dice", f"{float(nfc['best_val_dice']):.6f}")
                    q2.metric("NFC–ACO running best", f"{nfc_best:.6f}")
                    q3.metric("Random current Dice", f"{float(rnd['best_val_dice']):.6f}")
                    q4.metric("Random running best", f"{random_best:.6f}")

                chart_rows.append({
                    "Evaluation": i + 1,
                    "NFC–ACO": nfc_best,
                    "Random Search": random_best
                })
                chart_df = pd.DataFrame(chart_rows).set_index("Evaluation")
                chart_box.line_chart(chart_df, height=300)

                progress.progress((i + 1) / replay_budget)
                time.sleep(delay)

            best_nfc = nfc_run.loc[nfc_run["best_val_dice"].idxmax()]
            best_random = random_run.loc[random_run["best_val_dice"].idxmax()]

            status_box.success(
                f"✓ Recorded seed {replay_seed} replay complete — "
                f"{replay_budget} NFC–ACO and {replay_budget} Random Search "
                "candidate evaluations compared."
            )

            st.markdown("### Best Recorded Candidate for This Seed")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(
                    f"""
                    <div class="best">
                    <b>NFC–ACO</b><br>
                    Evaluation: <b>{int(best_nfc['evaluation_number'])}</b><br>
                    Learning rate: <b>{format_lr(best_nfc['learning_rate'])}</b><br>
                    Batch size: <b>{int(best_nfc['batch_size'])}</b><br>
                    Dropout: <b>{float(best_nfc['dropout']):.1f}</b><br>
                    Optimiser: <b>{str(best_nfc['optimizer']).upper()}</b><br>
                    Best proxy Dice: <b>{float(best_nfc['best_val_dice']):.6f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with b2:
                st.markdown(
                    f"""
                    <div class="research-card">
                    <b>Random Search</b><br>
                    Evaluation: <b>{int(best_random['evaluation_number'])}</b><br>
                    Learning rate: <b>{format_lr(best_random['learning_rate'])}</b><br>
                    Batch size: <b>{int(best_random['batch_size'])}</b><br>
                    Dropout: <b>{float(best_random['dropout']):.1f}</b><br>
                    Optimiser: <b>{str(best_random['optimizer']).upper()}</b><br>
                    Best proxy Dice: <b>{float(best_random['best_val_dice']):.6f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # STATIC EVIDENCE FROM ALL THREE RECORDED SEEDS
        # ----------------------------------------------------
        st.divider()
        st.subheader("Recorded Primary Experiment — All Three Seeds")

        summary_rows = []
        for seed in available_seeds:
            n = prepare_seed_run(hpo_df, seed, "NFC-ACO")
            r = prepare_seed_run(hpo_df, seed, "Random Search")
            if len(n) and len(r):
                summary_rows.append({
                    "Seed": seed,
                    "NFC–ACO best proxy Dice": n["best_val_dice"].max(),
                    "Random Search best proxy Dice": r["best_val_dice"].max(),
                    "NFC–ACO best evaluation": int(n.loc[n["best_val_dice"].idxmax(), "evaluation_number"]),
                    "Random best evaluation": int(r.loc[r["best_val_dice"].idxmax(), "evaluation_number"])
                })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style.format({
                "NFC–ACO best proxy Dice": "{:.6f}",
                "Random Search best proxy Dice": "{:.6f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        mean_nfc = summary_df["NFC–ACO best proxy Dice"].mean()
        mean_random = summary_df["Random Search best proxy Dice"].mean()

        s1, s2, s3 = st.columns(3)
        s1.metric("NFC–ACO mean best", f"{mean_nfc:.6f}")
        s2.metric("Random mean best", "0.487162")
        s3.metric("Absolute difference", "+0.033593")

        st.caption(
            "These values are calculated directly from FINAL_72_HPO_Evaluations.csv. "
            "The file contains the primary 12-candidate experiment only; the separate "
            "24-candidate supplementary experiment is therefore not replayed here."
        )

        # ----------------------------------------------------
        # MEAN RUNNING-BEST CONVERGENCE ACROSS THREE SEEDS
        # ----------------------------------------------------
        st.markdown("### Mean Search Convergence — Three Seeds")

        convergence_rows = []
        for seed in available_seeds:
            n = prepare_seed_run(hpo_df, seed, "NFC-ACO").copy()
            r = prepare_seed_run(hpo_df, seed, "Random Search").copy()

            if len(n) >= 12 and len(r) >= 12:
                n = n.iloc[:12].copy()
                r = r.iloc[:12].copy()
                n["running_best"] = n["best_val_dice"].cummax()
                r["running_best"] = r["best_val_dice"].cummax()

                for idx in range(12):
                    convergence_rows.append({
                        "Seed": seed,
                        "Evaluation": idx + 1,
                        "NFC–ACO": float(n.iloc[idx]["running_best"]),
                        "Random Search": float(r.iloc[idx]["running_best"])
                    })

        if convergence_rows:
            convergence_df = pd.DataFrame(convergence_rows)
            mean_convergence = (
                convergence_df
                .groupby("Evaluation")[["NFC–ACO", "Random Search"]]
                .mean()
            )
            # Convert to long form for a presentation-focused Vega-Lite chart.
            chart_df = (
                mean_convergence
                .reset_index()
                .melt(
                    id_vars="Evaluation",
                    value_vars=["NFC–ACO", "Random Search"],
                    var_name="Method",
                    value_name="Mean running-best Dice"
                )
            )

            # Final points are labelled so the examiner can read the result immediately.
            endpoint_df = chart_df[chart_df["Evaluation"] == 12].copy()
            endpoint_df["Label"] = endpoint_df["Mean running-best Dice"].map(
                lambda value: f"{value:.6f}"
            )

            convergence_spec = {
                "height": 360,
                "layer": [
                    {
                        "mark": {"type": "line", "point": True, "strokeWidth": 3},
                        "encoding": {
                            "x": {
                                "field": "Evaluation",
                                "type": "quantitative",
                                "title": "Candidate evaluation",
                                "scale": {"domain": [1, 12]},
                                "axis": {"values": list(range(1, 13)), "format": "d"}
                            },
                            "y": {
                                "field": "Mean running-best Dice",
                                "type": "quantitative",
                                "title": "Mean running-best validation Dice",
                                "scale": {"domain": [0.30, 0.55], "zero": False}
                            },
                            "color": {
                                "field": "Method",
                                "type": "nominal",
                                "title": "Method"
                            },
                            "tooltip": [
                                {"field": "Method", "type": "nominal"},
                                {"field": "Evaluation", "type": "quantitative", "format": "d"},
                                {"field": "Mean running-best Dice", "type": "quantitative", "format": ".6f"}
                            ]
                        }
                    },
                    {
                        "data": {"values": endpoint_df.to_dict(orient="records")},
                        "mark": {
                            "type": "text",
                            "align": "right",
                            "dx": -8,
                            "dy": -12,
                            "fontWeight": "bold",
                            "fontSize": 13
                        },
                        "encoding": {
                            "x": {"field": "Evaluation", "type": "quantitative"},
                            "y": {"field": "Mean running-best Dice", "type": "quantitative"},
                            "text": {"field": "Label", "type": "nominal"},
                            "color": {"field": "Method", "type": "nominal", "legend": None}
                        }
                    }
                ]
            }

            st.vega_lite_chart(
                chart_df,
                convergence_spec,
                use_container_width=True
            )
            st.caption(
                "Running-best validation Dice averaged across seeds 42, 123 and 2026. "
                "The x-axis shows exactly the shared 12-candidate evaluation budget; "
                "labels at evaluation 12 show the final mean running-best values."
            )

    st.divider()
    st.markdown(
        """
        <div class="best">
        <b>Configuration selected for extended U-Net++ training</b><br>
        Learning rate: <b>5 × 10⁻⁵</b> ·
        Batch size: <b>4</b> ·
        Dropout: <b>0.1</b> ·
        Optimiser: <b>RMSprop</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TAB 3 — EXPERIMENTAL EVIDENCE
# ============================================================

with tab3:

    st.header("Experimental Evidence")
    st.caption(
        "Results are presented in the same order as the experimental argument: "
        "baseline segmentation → restricted-budget HPO → extended final evaluation → runtime interpretation."
    )

    # --------------------------------------------------------
    # RESULT FLOW
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="flow">
        <div class="step">1 · Baseline Models</div><div class="arrow">→</div>
        <div class="step">2 · 12-Candidate Search</div><div class="arrow">→</div>
        <div class="step">3 · 24-Candidate Search</div><div class="arrow">→</div>
        <div class="step">4 · Final Model</div><div class="arrow">→</div>
        <div class="step">5 · Interpretation</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # 1. BASELINES
    # --------------------------------------------------------
    st.subheader("1. Baseline Segmentation Performance")
    b1, b2, b3 = st.columns(3)
    b1.metric("Baseline U-Net Dice", "0.9596")
    b2.metric("Baseline U-Net++ Dice", "0.9589")
    b3.metric("Absolute Dice difference", "0.0007")
    st.caption(
        "The two baseline architectures produced very similar Dice scores, "
        "providing the reference point for the later selected U-Net++ configuration."
    )

    # Compact horizontal comparison. AxisStart provides an explicit visual baseline
    # so the bars remain visible when the Dice axis is intentionally compressed.
    baseline_chart_df = pd.DataFrame({
        "Model": ["U-Net", "U-Net++", "NFC–ACO-selected U-Net++"],
        "Dice": [0.9596, 0.9589, 0.95612],
        "AxisStart": [0.954, 0.954, 0.954]
    })

    baseline_base = alt.Chart(baseline_chart_df).encode(
        y=alt.Y(
            "Model:N",
            title=None,
            sort=["U-Net", "U-Net++", "NFC–ACO-selected U-Net++"],
            axis=alt.Axis(labelLimit=260)
        ),
        tooltip=[
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("Dice:Q", title="Final test Dice", format=".5f")
        ]
    )

    baseline_bars = baseline_base.mark_bar(size=28, cornerRadiusEnd=5).encode(
        x=alt.X(
            "Dice:Q",
            title="Final test Dice",
            scale=alt.Scale(domain=[0.954, 0.961])
        ),
        x2="AxisStart:Q"
    )

    baseline_labels = baseline_base.mark_text(
        align="left",
        baseline="middle",
        dx=7,
        fontSize=14,
        fontWeight="bold"
    ).encode(
        x=alt.X("Dice:Q", scale=alt.Scale(domain=[0.954, 0.961])),
        text=alt.Text("Dice:Q", format=".5f")
    )

    baseline_chart = (baseline_bars + baseline_labels).properties(
        title="Final Test Dice Comparison",
        height=190
    )
    st.altair_chart(baseline_chart, use_container_width=True)
    st.caption(
        "The Dice axis starts at 0.954 only to make the small differences visible; "
        "the exact scores are printed beside each bar."
    )

    # --------------------------------------------------------
    # 2. PRIMARY SEARCH
    # --------------------------------------------------------
    st.divider()
    st.subheader("2. Primary Search — 12 Candidate Evaluations")

    p1, p2, p3 = st.columns(3)
    p1.metric("NFC–ACO Mean Best Proxy Dice", "0.520755")
    p2.metric("Random Search Mean Best Proxy Dice", "0.487162")
    p3.metric("Absolute Difference", "+0.033593")

    st.success(
        "Under the primary equal-budget experiment, NFC–ACO achieved the higher "
        "mean best proxy validation Dice across seeds 42, 123 and 2026."
    )
    st.caption(
        "Both methods used the same 48-configuration search space and the same "
        "12-candidate evaluation budget. Candidate-level replay is available in the NFC–ACO Search tab."
    )


    if hpo_df is not None:
        seed_rows = []
        for seed in [42, 123, 2026]:
            n = prepare_seed_run(hpo_df, seed, "NFC-ACO")
            r = prepare_seed_run(hpo_df, seed, "Random Search")
            if not n.empty and not r.empty:
                seed_rows.extend([
                    {"Seed": str(seed), "Method": "NFC–ACO", "Best Proxy Dice": float(n["best_val_dice"].max())},
                    {"Seed": str(seed), "Method": "Random Search", "Best Proxy Dice": float(r["best_val_dice"].max())}
                ])
        seed_chart_df = pd.DataFrame(seed_rows)
        if not seed_chart_df.empty:
            seed_chart_df["AxisStart"] = 0.45
            seed_base = alt.Chart(seed_chart_df).encode(
                x=alt.X("Seed:N", title="Random seed", sort=["42", "123", "2026"]),
                xOffset=alt.XOffset("Method:N", sort=["NFC–ACO", "Random Search"]),
                color=alt.Color("Method:N", title="Method"),
                tooltip=[
                    alt.Tooltip("Seed:N", title="Seed"),
                    alt.Tooltip("Method:N", title="Method"),
                    alt.Tooltip("Best Proxy Dice:Q", title="Best proxy Dice", format=".6f")
                ]
            )
            seed_bars = seed_base.mark_bar(size=42).encode(
                y=alt.Y(
                    "Best Proxy Dice:Q",
                    title="Best proxy validation Dice",
                    scale=alt.Scale(domain=[0.45, 0.55])
                ),
                y2="AxisStart:Q"
            )
            seed_labels = seed_base.mark_text(dy=-9, fontSize=12).encode(
                y=alt.Y("Best Proxy Dice:Q", scale=alt.Scale(domain=[0.45, 0.55])),
                text=alt.Text("Best Proxy Dice:Q", format=".3f")
            )
            seed_chart = (seed_bars + seed_labels).properties(
                title="Primary Experiment — Best Proxy Dice by Seed",
                height=300
            )
            st.altair_chart(seed_chart, use_container_width=True)

    # --------------------------------------------------------
    # 3. SUPPLEMENTARY SEARCH
    # --------------------------------------------------------
    st.divider()
    st.subheader("3. Supplementary Search — 24 Candidate Evaluations")

    s1, s2, s3 = st.columns(3)
    s1.metric("NFC–ACO Mean Best Proxy Dice", "0.530415")
    s2.metric("Random Search Mean Best Proxy Dice", "0.468900")
    s3.metric("Absolute Difference", "+0.061515")

    st.info(
        "The 24-candidate experiment was a fresh supplementary optimisation experiment, "
        "not a continuation of the original 12-candidate runs. It provides a second "
        "restricted-budget comparison of search-stage behaviour."
    )


    budget_chart_df = pd.DataFrame({
        "Budget": ["12 candidates", "12 candidates", "24 candidates", "24 candidates"],
        "Method": ["NFC–ACO", "Random Search", "NFC–ACO", "Random Search"],
        "Mean Best Proxy Dice": [0.520755, 0.487162, 0.530415, 0.468900],
        "AxisStart": [0.45, 0.45, 0.45, 0.45]
    })
    budget_base = alt.Chart(budget_chart_df).encode(
        x=alt.X(
            "Budget:N",
            title="Candidate-evaluation budget",
            sort=["12 candidates", "24 candidates"]
        ),
        xOffset=alt.XOffset("Method:N", sort=["NFC–ACO", "Random Search"]),
        color=alt.Color("Method:N", title="Method"),
        tooltip=[
            alt.Tooltip("Budget:N", title="Budget"),
            alt.Tooltip("Method:N", title="Method"),
            alt.Tooltip("Mean Best Proxy Dice:Q", title="Mean best proxy Dice", format=".6f")
        ]
    )
    budget_bars = budget_base.mark_bar(size=55).encode(
        y=alt.Y(
            "Mean Best Proxy Dice:Q",
            title="Mean best proxy Dice",
            scale=alt.Scale(domain=[0.45, 0.55])
        ),
        y2="AxisStart:Q"
    )
    budget_labels = budget_base.mark_text(dy=-9, fontSize=12).encode(
        y=alt.Y("Mean Best Proxy Dice:Q", scale=alt.Scale(domain=[0.45, 0.55])),
        text=alt.Text("Mean Best Proxy Dice:Q", format=".3f")
    )
    budget_chart = (budget_bars + budget_labels).properties(
        title="Evaluation Budget Comparison — 12 vs 24 Candidates",
        height=300
    )
    st.altair_chart(budget_chart, use_container_width=True)
    st.caption("The 24-candidate bars use the reported aggregate results only; no unrecorded candidate-level trajectory is inferred.")

    # --------------------------------------------------------
    # 4. FINAL SEGMENTATION
    # --------------------------------------------------------
    st.divider()
    st.subheader("4. Extended Selected Model Evaluation")

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Accuracy", "0.97842")
    f2.metric("Dice", "0.95612")
    f3.metric("IoU", "0.91593")
    f4.metric("Precision", "0.96307")
    f5.metric("Recall", "0.94926")

    st.markdown("#### Final Dice comparison")
    q1, q2, q3 = st.columns(3)
    q1.metric("Baseline U-Net++", "0.95890")
    q2.metric("NFC–ACO-selected U-Net++", "0.95612")
    q3.metric("Selected − Baseline", "−0.00278")

    st.warning(
        "Key interpretation: stronger low-fidelity proxy-search performance did not "
        "automatically translate into higher final segmentation Dice after extended training. "
        "The selected model reached 0.95612 compared with 0.9589 for the baseline U-Net++."
    )

    # --------------------------------------------------------
    # 5. RUNTIME
    # --------------------------------------------------------
    st.divider()
    st.subheader("5. Primary Search Runtime")

    r1, r2, r3 = st.columns(3)
    r1.metric("NFC–ACO", "≈ 10.55 min")
    r2.metric("Random Search", "≈ 10.07 min")
    r3.metric("Approx. Difference", "+0.48 min")

    st.caption(
        "The primary experiment did not demonstrate a wall-clock runtime advantage for NFC–ACO; "
        "runtime measurements are environment-dependent."
    )

    # --------------------------------------------------------
    # EXAMINER TAKEAWAY
    # --------------------------------------------------------
    st.divider()
    st.subheader("Research Takeaway")
    st.markdown(
        """
        <div class="best">
        <b>Search-stage result:</b> NFC–ACO produced higher mean best proxy Dice than Random Search
        under both tested restricted candidate budgets.<br><br>
        <b>Final-model result:</b> the selected U-Net++ configuration did not exceed the baseline
        U-Net++ final Dice.<br><br>
        <b>Meaning:</b> the experiment supports a distinction between improved low-fidelity
        candidate search and improved final segmentation performance.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TAB 4 — RESEARCH SUMMARY
# ============================================================

with tab4:

    st.header("Research Summary")
    st.caption(
        "Examiner-facing overview of the research question, experimental design, "
        "main evidence, contribution and limitations."
    )

    # --------------------------------------------------------
    # 1. AIM
    # --------------------------------------------------------
    st.subheader("1. Research Aim")
    st.info(
        "To investigate whether an NFC–ACO-inspired hyperparameter optimisation "
        "strategy can identify competitive U-Net++ configurations for lung "
        "segmentation under restricted candidate-evaluation budgets."
    )

    # --------------------------------------------------------
    # 2. DATASET AND SPLIT
    # --------------------------------------------------------
    st.subheader("2. Dataset and Experimental Split")

    d1, d2, d3 = st.columns(3)
    d1.metric("Total paired images", "704")
    d2.metric("Montgomery", "138")
    d3.metric("Shenzhen", "566")

    s1, s2, s3 = st.columns(3)
    s1.metric("Training", "492")
    s2.metric("Validation", "106")
    s3.metric("Test", "106")

    st.caption(
        "Chest X-rays and corresponding lung masks were preprocessed to 256 × 256."
    )

    # --------------------------------------------------------
    # 3. METHODOLOGY
    # --------------------------------------------------------
    st.subheader("3. Methodology at a Glance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Search space", "48 configs")
    m2.metric("Primary budget", "12 candidates")
    m3.metric("Supplementary budget", "24 candidates")
    m4.metric("Optimisation seeds", "3")

    st.markdown(
        "**Workflow:** U-Net++ → NFC-inspired exploration → low-fidelity candidate "
        "evaluation → validation Dice fitness → ACO-style reinforcement → selected "
        "configuration → extended final training and test evaluation."
    )

    st.caption(
        "Low-fidelity candidate evaluation used 180 training images, 80 validation "
        "images, a maximum of 5 epochs and early-stopping patience of 2."
    )

    # --------------------------------------------------------
    # 4. KEY RESULTS
    # --------------------------------------------------------
    st.subheader("4. Key Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("12-candidate NFC–ACO", "0.520755")
    r2.metric("12-candidate Random", "0.487162")
    r3.metric("24-candidate NFC–ACO", "0.530415")
    r4.metric("24-candidate Random", "0.468900")

    f1, f2, f3 = st.columns(3)
    f1.metric("Baseline U-Net Dice", "0.9596")
    f2.metric("Baseline U-Net++ Dice", "0.9589")
    f3.metric("Selected final Dice", "0.95612", delta="-0.00278 vs U-Net++")

    st.success(
        "Main finding: NFC–ACO achieved higher mean best proxy Dice than Random "
        "Search under both tested restricted budgets. However, the selected "
        "configuration did not exceed the baseline U-Net++ final Dice after "
        "extended training."
    )

    # --------------------------------------------------------
    # 5. SELECTED CONFIGURATION
    # --------------------------------------------------------
    st.subheader("5. Selected Configuration")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Learning rate", "5 × 10⁻⁵")
    c2.metric("Batch size", "4")
    c3.metric("Dropout", "0.1")
    c4.metric("Optimiser", "RMSprop")

    # --------------------------------------------------------
    # 6. CONTRIBUTION
    # --------------------------------------------------------
    st.subheader("6. Research Contribution")
    st.write(
        "The study provides a controlled equal-budget investigation of an "
        "NFC-inspired exploration and ACO-style reinforcement strategy for "
        "discrete U-Net++ hyperparameter optimisation. The results explicitly "
        "distinguish search-stage proxy performance from final segmentation "
        "performance rather than claiming universal superiority."
    )

    # --------------------------------------------------------
    # 7. LIMITATIONS AND FUTURE WORK
    # --------------------------------------------------------
    st.subheader("7. Limitations and Future Work")

    left, right = st.columns(2)

    with left:
        st.markdown("**Limitations**")
        st.markdown(
            """• Three optimisation seeds.  
• Restricted 48-configuration search space.  
• Low-fidelity proxy candidate evaluation.  
• 704 paired images with no external validation dataset.  
• Runtime measurements are environment-dependent."""
        )

    with right:
        st.markdown("**Future work**")
        st.markdown(
            """• Evaluate more random seeds and larger search spaces.  
• Investigate multi-fidelity optimisation.  
• Perform external-dataset validation.  
• Compare additional HPO baselines including Bayesian optimisation, Hyperband and BOHB."""
        )

    st.divider()
    st.markdown(
        """### One-sentence conclusion
**NFC–ACO improved the quality of the best low-fidelity candidates found under the tested budgets, but this search-stage advantage did not translate into a higher final segmentation Dice than the baseline U-Net++.**"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MSc Artificial Intelligence Dissertation Demonstrator | "
    "NFC–ACO-Inspired Hyperparameter Optimisation of U-Net++ "
    "for Lung Segmentation in Chest X-ray Images"
)