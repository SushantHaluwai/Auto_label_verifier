import streamlit as st
import requests
from PIL import Image, ImageDraw
import pandas as pd

st.set_page_config(layout="wide")
st.title("🔍 AI Label Verification System")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded:

    col1, col2 = st.columns([2, 1])

    image = Image.open(uploaded).convert("RGB")

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Run Verification"):

        with st.spinner("Running ensemble + critique..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            response = requests.post("http://localhost:8000/process", files=files)

        if response.status_code == 200:

            result = response.json()
            detections = result["detections"]

            # Sidebar Controls
            st.sidebar.header("Display Options")
            min_conf = st.sidebar.slider(
                "Minimum Confidence",
                0.0, 1.0, 0.5, 0.05
            )

            show_yolo = st.sidebar.checkbox("Show YOLO", True)
            show_rcnn = st.sidebar.checkbox("Show RCNN", True)

            filtered = []
            for d in detections:
                if d["calibrated_score"] < min_conf:
                    continue
                if d["model"] == "yolo" and not show_yolo:
                    continue
                if d["model"] == "rcnn" and not show_rcnn:
                    continue
                filtered.append(d)

            # Draw filtered detections
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)

            label_map = {
                0: "person", 1: "bicycle", 2: "car",
                3: "motorcycle", 4: "airplane",
                5: "bus", 6: "train",
                7: "truck", 8: "boat"
            }

            color_map = {"yolo": "red", "rcnn": "blue"}

            for det in filtered:
                bbox = det["bbox"]
                score = det["calibrated_score"]
                cls = det["class"]
                model_name = det["model"]

                label = label_map.get(cls, f"class_{cls}")
                color = color_map.get(model_name, "green")

                draw.rectangle(bbox, outline=color, width=4)
                draw.text(
                    (bbox[0], max(0, bbox[1] - 20)),
                    f"{label} | {score:.2f}",
                    fill=color
                )

            with col1:
                st.image(image_copy, caption="Fused Detections", use_column_width=True)

            # 🔥 Right Panel Info
            with col2:
                st.subheader("Status")

                if result["status"] == "approved":
                    st.success(" Approved")
                else:
                    st.error("⚠ Needs Human Review")

                st.subheader("AI Critique")
                st.info(result["critique"])

                st.subheader("Detection Summary")

                if filtered:
                    df = pd.DataFrame(filtered)
                    st.dataframe(df)
                else:
                    st.warning("No detections above selected threshold.")

        else:
            st.error("API Error")
            st.write(response.text)
