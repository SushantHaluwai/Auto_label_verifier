import streamlit as st
import requests
from PIL import Image, ImageDraw

st.set_page_config(layout="wide")
st.title("🧠 Model Comparison Dashboard")

uploaded = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    # -------------------------
    # Sidebar Controls
    # -------------------------
    st.sidebar.header("Controls")

    min_conf = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

    view_mode = st.sidebar.radio(
        "View Mode",
        ["Raw", "Calibrated"]
    )

    overlay_mode = st.sidebar.checkbox("Overlay Models", False)

    if st.button("Run Models"):

        with st.spinner("Running inference..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            response = requests.post("http://api:8000/process", files=files)

        if response.status_code == 200:

            result = response.json()

            if view_mode == "Raw":
                detections = result["raw_detections"]
            else:
                detections = result["calibrated_detections"]

            models = list(detections.keys())

            color_map = {
                "rcnn": "blue",
                "yolo": "red"
            }

            # =============================
            # Overlay Mode
            # =============================
            if overlay_mode:

                img_copy = image.copy()
                draw = ImageDraw.Draw(img_copy)

                for model in models:
                    for det in detections[model]:

                        score = det["score"] if view_mode == "Raw" else det["calibrated_score"]

                        if score < min_conf:
                            continue

                        bbox = det["bbox"]
                        category = det["category"]

                        color = color_map.get(model, "green")

                        draw.rectangle(bbox, outline=color, width=3)
                        draw.text(
                            (bbox[0], max(0, bbox[1] - 20)),
                            f"{category} | {score:.2f}",
                            fill=color
                        )

                st.image(img_copy, caption="Overlay View", width="stretch")

            # =============================
            # Side-by-Side Mode
            # =============================
            else:

                cols = st.columns(len(models))

                for idx, model in enumerate(models):

                    img_copy = image.copy()
                    draw = ImageDraw.Draw(img_copy)

                    for det in detections[model]:

                        score = det["score"] if view_mode == "Raw" else det["calibrated_score"]

                        if score < min_conf:
                            continue

                        bbox = det["bbox"]
                        category = det["category"]

                        color = color_map.get(model, "green")

                        draw.rectangle(bbox, outline=color, width=3)
                        draw.text(
                            (bbox[0], max(0, bbox[1] - 20)),
                            f"{category} | {score:.2f}",
                            fill=color
                        )

                    cols[idx].image(
                        img_copy,
                        caption=model.upper(),
                        width="stretch"
                    )

            # -------------------------
            # Detection Count Summary
            # -------------------------
            st.subheader("📊 Detection Summary")

            for model in models:
                total = sum(
                    1 for d in detections[model]
                    if (d["score"] if view_mode == "Raw" else d["calibrated_score"]) >= min_conf
                )
                st.write(f"**{model.upper()}** → {total} detections")

        else:
            st.error("API Error")