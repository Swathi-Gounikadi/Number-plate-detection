import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import easyocr


st.set_page_config(
    page_title="Number Plate Detection using YOLO",
    page_icon="🚗",
    layout="centered"
)

st.title("Car Number Plate Detection System")

st.write("Upload an **image or video** to detect **car number plates and read plate numbers using EasyOCR**.")


@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


@st.cache_resource
def load_ocr():
    reader = easyocr.Reader(['en'])
    return reader

reader = load_ocr()


confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

source_type = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video"]
)



if source_type == "Image":

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Detect Number Plate"):

            results = model.predict(
                source=image_np,
                conf=confidence,
                save=False
            )

            annotated_frame = results[0].plot()

            st.image(annotated_frame, caption="Detection Result", use_column_width=True)

            boxes = results[0].boxes.xyxy.cpu().numpy()

            plate_texts = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                plate_crop = image_np[y1:y2, x1:x2]

                ocr_result = reader.readtext(plate_crop)

                for detection in ocr_result:
                    text = detection[1]
                    plate_texts.append(text)

            if plate_texts:
                st.success("Detected Plate Number(s):")

                for plate in plate_texts:
                    st.write("🚗", plate)
            else:
                st.warning("No plate text detected")


elif source_type == "Video":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        st.write("Processing video...")

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=confidence,
                save=False
            )

            annotated_frame = results[0].plot()

            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                plate_crop = frame[y1:y2, x1:x2]

                ocr_result = reader.readtext(plate_crop)

                for detection in ocr_result:
                    text = detection[1]

                    cv2.putText(
                        annotated_frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0),
                        2
                    )

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_frame, use_column_width=True)

        cap.release()

        st.success("Video processing completed")