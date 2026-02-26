import streamlit as st
import requests
from PIL import Image, ImageDraw

st.set_page_config(layout="wide")
st.title("MoE Detection Dashboard")

uploaded = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.sidebar.header("Controls")

    min_conf = st.sidebar.slider(
        "Minimum Confidence",
        0.0, 1.0, 0.3, 0.05
    )

    mode = st.sidebar.selectbox(
        "Detection Mode",
        [
            "Raw",
            "Raw Refined (Single)",
            "Raw Refined (Cross)",
            "Calibrated",
            "Calibrated Refined (Single)",
            "Calibrated Refined (Cross)"
        ]
    )

    if st.button("Run Models"):

        with st.spinner("Running inference..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            response = requests.post("http://api:8000/process", files=files)

        if response.status_code != 200:
            st.error("API Error")
            st.stop()

        result = response.json()

        # ---------------------------------
        # Select Output Mode
        # ---------------------------------

        if mode == "Raw":
            detections = result["raw"]

        elif mode == "Raw Refined (Single)":
            detections = result["raw_refined_single"]

        elif mode == "Raw Refined (Cross)":
            detections = {"ensemble": result["raw_refined_cross"]}

        elif mode == "Calibrated":
            detections = result["calibrated"]

        elif mode == "Calibrated Refined (Single)":
            detections = result["calibrated_refined_single"]

        else:
            detections = {"ensemble": result["calibrated_refined_cross"]}

        # ---------------------------------
        # Draw Results
        # ---------------------------------

        if isinstance(detections, dict):

            cols = st.columns(len(detections))

            for idx, (model_name, det_list) in enumerate(detections.items()):

                img_copy = image.copy()
                draw = ImageDraw.Draw(img_copy)

                for det in det_list:

                    score = det.get("calibrated_score", det.get("score"))

                    if score < min_conf:
                        continue

                    bbox = det["bbox"]

                    draw.rectangle(bbox, outline="red", width=3)

                    contributors = det.get("contributors", [])

                    text = f"{det['category']} | {score:.2f}"

                    if contributors:
                        text += f" | {','.join(contributors)}"

                    draw.text(
                        (bbox[0], max(0, bbox[1] - 20)),
                        text,
                        fill="red"
                    )

                cols[idx].image(img_copy, caption=model_name, use_container_width=True)

        else:
            # Cross-model list case
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)

            for det in detections:

                score = det.get("calibrated_score", det.get("score"))

                if score < min_conf:
                    continue

                bbox = det["bbox"]
                contributors = det.get("contributors", [])

                draw.rectangle(bbox, outline="blue", width=3)

                text = f"{det['category']} | {score:.2f}"

                if contributors:
                    text += f" | {','.join(contributors)}"

                draw.text(
                    (bbox[0], max(0, bbox[1] - 20)),
                    text,
                    fill="blue"
                )

            st.image(img_copy, caption="Ensemble Output", use_container_width=True)