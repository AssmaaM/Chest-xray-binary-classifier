
<img width="300" alt="Xray" src="https://github.com/user-attachments/assets/9216e15e-86b7-414e-96f4-914e6742a0bb" />

This project implements a binary image classifier using Keras and TensorFlow to detect lung opacity from chest X-ray images using two approachs :
1-buuliding from scratch a CNN model 
2-using VGG16 pre-trained model and upload it to a streamlit app to test it

  ----- Classes -----
- Normal: No signs of opacity.
- Opacity: Lung opacities visible

  ----- Architecture -----

- 3 × Conv2D + MaxPooling2D layers
- Flatten layer
- Dense layer 
- Output layer (1 neuron, sigmoid activation)

  ----- Input Format -----

- Input: Chest X-ray image
- Preprocessing: Resized to 100×100 pixels, normalized (pixel values / 255)

  ----- Output -----

- 0 → Normal
- 1 → Opacity detected

  ----- Dependencies -----

- TensorFlow / Keras
- NumPy
- OpenCV (for image preprocessing)
- Matplotlib
- streamlit

Install with:

pip install tensorflow numpy opencv-python matplotlib streamlit

  ----- Dataset -----

Download the datasets from (https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images?select=val)

  ----- Run the app using anaconda prompt ------

- browse into the python file app directory 
- run the following command streamlit run XrayClassifierStremlit.py
