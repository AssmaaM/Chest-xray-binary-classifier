This project implements a binary image classifier using Keras and TensorFlow to detect lung opacity from chest X-ray images

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

Install with:

pip install tensorflow numpy opencv-python matplotlib
