
# Face Recognition App

This is a **Face Recognition App** built using **Streamlit**, **OpenCV**, and **Face Recognition** libraries. The app allows users to upload images to add new faces, save and load known faces, and perform live face recognition using a webcam.

## Features

- **Add New Faces**: Upload images of new people and associate them with their names.
- **Save/Load Faces**: Save the face encodings of known faces and load them when restarting the app.
- **Live Face Recognition**: Start a live webcam session where the app recognizes faces in real-time.
- **Face Detection**: Displays detected faces with rectangles around them, and the recognized names are shown under the detected faces.

## Requirements

To run the app, you need the following libraries:

- `streamlit`
- `opencv-python`
- `face_recognition`
- `numpy`
- `Pillow`
- `pickle` (for saving/loading face encodings)

You can install the required packages using the following command:

```bash
pip install streamlit opencv-python face_recognition numpy Pillow
```

## How to Run the App

1. **Clone the Repository** (or download the script):

```bash
git clone https://github.com/your-repo/face-recognition-app.git
cd face-recognition-app
```

2. **Install Dependencies**:

Make sure to install all the necessary dependencies using `pip` as mentioned in the "Requirements" section.

3. **Run the App**:

Use the following command to start the app:

```bash
streamlit run app.py
```

4. **Usage**:

- Open the app in your browser using the URL displayed in the terminal (`http://localhost:8501`).
- **Add New Faces**: Upload a photo and enter the name in the sidebar to add a new face.
- **Start Webcam**: Click the "Start Webcam for Face Recognition" button to begin real-time face recognition.
- **Stop Webcam**: Click "Stop Webcam" to end the session and release the camera.

## Directory Structure

```
.
├── app.py               # Main app file
├── known_faces.pkl       # Encoded faces stored in this file (after adding faces)
└── README.md             # This README file
```

## Important Notes

- **Webcam Access**: Make sure to give permission to access your webcam when using the live recognition feature.
- **Face Detection Limitations**: The app might struggle with blurry or low-resolution images. Ensure good lighting conditions for better recognition accuracy.
- **Known Faces Storage**: The app uses a file named `known_faces.pkl` to store face encodings. This file is saved in the root directory and is loaded automatically when the app starts.

## Troubleshooting

- **Webcam Issues**: If the webcam doesn't start, ensure that no other application is using it and that the browser has permission to access the camera.
- **No Faces Detected**: If faces are not being detected in the uploaded image, ensure that the image quality is good and contains a clear frontal face.

## Future Enhancements

- Add a feature to update or remove known faces.
- Implement user authentication to manage face data securely.
- Improve performance by skipping frames during webcam recognition.

## Acknowledgements

- The app is built using the **Face Recognition** library by [Adam Geitgey](https://github.com/ageitgey/face_recognition).
- **Streamlit** is used for the web interface, making it easy to deploy machine learning models with minimal setup.
