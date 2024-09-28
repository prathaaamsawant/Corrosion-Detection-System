import streamlit as st #UI
import os           
from PIL import Image
from roboflow import Roboflow
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace().project("dataset-corrosao-faca2")
model = project.version(1).model

# Define the Streamlit app
def main():
    st.title("IMAGE PROCESSING OF CORROSION IN STEEL STRUCTURES")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Perform inference and display annotated image
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform inference
        try:
            # Save uploaded file to a temporary location
            temp_file_path = "temp_image.jpg"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Perform inference with the file path
            result = model.predict(temp_file_path).json()
            annotated_image_path = "annotated_image.jpg"
            # Save annotated image
            model.predict(temp_file_path).save(annotated_image_path)
            st.success("Image annotated successfully!")
            st.image(annotated_image_path, caption="Annotated Image", use_column_width=True)

            # Load the two images
            image1 = Image.open('temp_image.jpg')
            image2 = Image.open('annotated_image.jpg')

            # Convert the images to numpy arrays
            image1_array = np.array(image1)
            image2_array = np.array(image2)

            # calc the difference between the two images on basis of pixel values and ignore black pixels
            diff = np.abs(image1_array - image2_array)
            diff[diff < 10] = 0

            # Convert the difference array to pixels
            diff_image = Image.fromarray(diff.astype(np.uint8))

            # calculate the percentage of difference between difference image and original image
            total_pixels = image1_array.size
            print(total_pixels)
            diff_pixels = np.count_nonzero(diff)
            print(diff_pixels)

            corrosion = total_pixels - diff_pixels
            corrosion = corrosion / 10000
            if corrosion > 100 :
                corrosion = 90
            print(corrosion)

            st.success(f"Percent of Corrosion: {corrosion} %")

            if corrosion < 30:
                st.success("90% probability of no corrosion")
            elif 40 > corrosion > 60:
                st.success("The object can be reused by applying treatment to the corroded area")
            else:
                    st.error("90% probability of corrosion")


            # Delete temporary file
            os.remove(temp_file_path)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
