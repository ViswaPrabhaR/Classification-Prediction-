
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import numpy as np


import cv2
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib import pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def app():

    st.markdown("### Upload a Image")
    with st.container(border=True):
        uploaded_image = st.file_uploader("upload here", label_visibility="collapsed", type=["png", "jpeg", "jpg"])

    if uploaded_image is not None:
            st.header("Image Pre-Processing - Open CV")
            radio = st.radio('', options=['Varying Hues', 'Rotation','Noise Reduce', 'Dilation','Erosion','Threshold', 'Edge Detection'],
              horizontal=True)
            with st.container(border=True):
                original,preprocess=st.columns(2)
                #st.markdown("""<hr style="height:5px;border:none;color:red ;background-color:#333;" /> """, unsafe_allow_html=True)
                image = cv2.imread(f"data/Image_Process/{uploaded_image.name}")
                gray_image = cv2.imread(f"data/Image_Process/temp/bw_{uploaded_image.name}")
                ocr_result = pytesseract.image_to_string(image)
                c1, c2  = st.columns(2)
                c4, c5  = st.columns(2)

                with original:
                    with c1:
                        st.write(":green[*Original Image*]")
                    with c4:
                        st.image(uploaded_image)

                with preprocess:
                    if radio == "Varying Hues":
                        with c2:
                            color=st.radio(":green[*Select Hues*] ", options=['Grayscale','Inverting','HSV'],horizontal=True)
                        with c5:
                            if color == "HSV":
                                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                                st.image(hsv_image)
                            if color == "Inverting":
                                invert_image = cv2.bitwise_not(image)
                                st.image(invert_image)
                            if color == "Grayscale":
                                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                cv2.imwrite(f"data/Image_Process/temp/bw_{uploaded_image.name}", gray_image)
                                st.image(gray_image)

                    if radio == "Rotation":
                        with c2:
                            angle = st.slider(':green[*Select Angle*]', 0, 360, 30)
                        with c5:

                            # Convert BGR image to RGB
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            # Image rotation parameter
                            center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
                            scale = 1

                            # getRotationMatrix2D creates a matrix needed for transformation.
                            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

                            # We want matrix for rotation w.r.t center to 30 degree without scaling.
                            rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (image.shape[1], image.shape[0]))
                            st.image(rotated_image)

                    if radio == "Noise Reduce":
                        with c2:
                            filter = st.radio(":green[*Select Image Filtering*] ", options=['Bilateral','Blur', 'Gaussian','Median'],
                                             horizontal=True)
                        with c5:
                            if filter == "Blur":
                                blur_image = cv2.blur(image, (7, 7))
                                st.image(blur_image)
                            if filter == "Bilateral":
                                bilateral = cv2.bilateralFilter(image, 15, 75, 75)
                                st.image(bilateral)
                            if filter == "Gaussian":
                                gaussian = cv2.GaussianBlur(image, (5, 5), 2)
                                st.image(gaussian)
                            if filter == "Median":
                                median = cv2.medianBlur(image,5)
                                st.image(median)

                    if radio == "Dilation":
                        with c2:
                            dilated_range = st.slider(':green[*Select Range*]', 0, 20, 8)
                        with c5:
                            kernel = np.ones((dilated_range, dilated_range), np.uint8)
                            dilated_image = cv2.dilate(gray_image,kernel)
                            st.image(dilated_image)

                    if radio == "Erosion":
                        with c2:
                            eroded_range = st.slider(':green[*Select Range*]', 0, 20, 8)
                        with c5:
                            kernel = np.ones((eroded_range, eroded_range), np.uint8)
                            eroded_image = cv2.erode(gray_image, kernel)
                            st.image(eroded_image)

                    if radio == "Threshold":
                        with c2:
                            st.write(":green[*Threshold Image*]")
                        with c5:
                            var, threshold_image = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
                            st.image(threshold_image)

                    if radio == "Edge Detection":
                        with c2:
                            st.write(":green[*Edge Detection*]")
                        with c5:
                            image_edge = cv2.Canny(gray_image, 50, 50)
                            st.image(image_edge)

                if ocr_result is not None:
                    with st.expander("Text Content"):
                        st.code(f"""{ocr_result}""")

