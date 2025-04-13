from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget, QFileDialog, QScrollArea, QFrame, QHBoxLayout
)
import tensorflow as tf
import numpy as np
import cv2
import os
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import sys
from main_image_processing import main_func
from OCR.tamil_ocr import get_tamil
from OCR.sanskrit_ocr import get_sanskrit

from AI_Interpretation.sample import get_response


import tensorflow as tf
import numpy as np
import cv2
import os

# ✅ Load the trained model
model = tf.keras.models.load_model("C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/sanskrit_tamil_classifier.h5")


# ✅ Function to preprocess an image
def preprocess_image(image_path):
    """
    Load and preprocess the input image for model prediction.
    - Converts image to grayscale
    - Resizes it to 224x224 (same as training data)
    - Normalizes pixel values to [0, 1]
    - Expands dimensions to match model input shape
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=[0, -1])  # Expand dimensions to match (1, 224, 224, 1)
    return img

# ✅ Function to make predictions
def predict_language(image_path):
    """
    Predicts the language of a given manuscript image.
    - Outputs "Sanskrit" or "Tamil" with confidence score.
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]  # Get prediction score
    print(prediction)
    # Convert to class label (Assuming 0 = Sanskrit, 1 = Tamil)
    predicted_class = "Tamil" if prediction > 0.5 else "Sanskrit"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Predicted Language: {predicted_class} (Confidence: {confidence:.2f})")
    return predicted_class

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Processing App")
        self.setGeometry(800, 800, 2000, 800)  # Slightly wider layout

        # Scroll area to make UI more accessible
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        # Main container widget inside scroll area
        self.scroll_widget = QWidget()
        self.layout = QVBoxLayout(self.scroll_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        # Upload button
        self.upload_btn = QPushButton("📂 Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setObjectName("uploadButton")

        # Image display section
        self.image_frame = QFrame()
        self.image_layout = QVBoxLayout(self.image_frame)

        # Input Image
        self.image_label = QLabel("Input Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("imageLabel")

        # Processed Image
        self.processed_label = QLabel("Processed Image")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setObjectName("imageLabel")

        self.image_layout.addWidget(self.image_label)
        self.image_layout.addWidget(self.processed_label)

        # Language Label
        self.lang_label = QLabel("Language: Tamil")
        self.lang_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.lang_label.setAlignment(Qt.AlignCenter)

        # OCR Output
        self.ocr_output_label = QLabel("OCR Output:")
        self.ocr_output_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.ocr_output_text = QTextEdit()
        self.ocr_output_text.setPlaceholderText("Detected text will appear here...")
        self.ocr_output_text.setReadOnly(True)
        self.ocr_output_text.setObjectName("textArea")

        # AI Interpretation Output
        self.ai_output_label = QLabel("AI Interpretation:")
        self.ai_output_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.ai_output_text = QTextEdit()
        self.ai_output_text.setPlaceholderText("AI-generated text will appear here...")
        self.ai_output_text.setReadOnly(True)
        self.ai_output_text.setObjectName("textArea")

        # Process Button
        self.process_btn = QPushButton("⚙️ Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setObjectName("processButton")

        # Display Step Outputs Button
        self.step_outputs_btn = QPushButton("📸 Display Step Outputs")
        self.step_outputs_btn.clicked.connect(self.display_step_outputs)
        self.step_outputs_btn.setObjectName("processButton")

        # Step Output Images Layout
        self.step_output_layout = QVBoxLayout()

        # Add widgets to layout
        self.layout.addWidget(self.upload_btn)
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.lang_label)
        self.layout.addWidget(self.ocr_output_label)
        self.layout.addWidget(self.ocr_output_text)
        self.layout.addWidget(self.ai_output_label)
        self.layout.addWidget(self.ai_output_text)
        self.layout.addWidget(self.process_btn)
        self.layout.addWidget(self.step_outputs_btn)
        self.layout.addLayout(self.step_output_layout)

        # Set layout inside scroll widget
        self.scroll_area.setWidget(self.scroll_widget)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

        # Apply stylesheet for a modern UI
        self.setStyleSheet(self.get_stylesheet())

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.file_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()

    def process_image(self):
        if hasattr(self, 'file_path'):
            # Step 1: Process Image
            output_path = main_func(self.file_path)
            output_pixmap = QPixmap(output_path)

            # Ensure output image matches input image size
            input_pixmap = self.image_label.pixmap()
            if input_pixmap:
                output_pixmap = output_pixmap.scaled(input_pixmap.width(), input_pixmap.height(), Qt.KeepAspectRatio)

            self.processed_label.setPixmap(output_pixmap)
            self.processed_label.adjustSize()
            lang=predict_language(output_path)
            # Step 2: Language Detection (Hardcoded for now)
            self.lang_label.setText(f"Language: {lang}")
            if lang=="Sanskrit":
            # Step 3: OCR Processing
                detected_text = get_sanskrit(output_path)
            if lang=="Tamil":
                detected_text = get_tamil(output_path)
            self.ocr_output_text.setText(detected_text)

            # Step 4: AI Interpretation
            ai_response = get_response(detected_text,lang)
            self.ai_output_text.setText(ai_response)

        else:
            self.ocr_output_text.setText("No image uploaded.")
            self.ai_output_text.setText("No image uploaded.")

    def display_step_outputs(self):
        step_labels = [
            "Grayscale Image", "Wavelet Denoised", "CLAHE Enhanced", "ESRGAN Enhanced",
            "Background Separated", "Sauvola Output", "Unsharp Masking", "OCR Specifics"
        ]
        step_paths = [
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/1_grayscale.png",
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/2_wavelet_denoised.png",
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/3_clahe_enhanced.png",
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/4_esrgan_enhanced.png",
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/5_background_seperated.png",
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/6_savuola_output.png",
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/7_unsharp_masking_output.png",
            "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/final_output/output.png"
        ]

        for i in range(len(step_paths)):
            label = QLabel(step_labels[i])
            label.setAlignment(Qt.AlignCenter)
            image_label = QLabel()
            pixmap = QPixmap(step_paths[i])
            if self.image_label.pixmap():
                pixmap = pixmap.scaled(self.image_label.pixmap().size(), Qt.KeepAspectRatio)
            image_label.setPixmap(pixmap)
            image_label.adjustSize()
            self.step_output_layout.addWidget(label)
            self.step_output_layout.addWidget(image_label)
    def get_stylesheet(self):
        return """
        QWidget {
            background-color: #FFFFFF;
        }
        QPushButton {
            background-color: #0078D7;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            border: none;
        }
        QPushButton:hover {
            background-color: #005A9E;
        }
        QLabel {
            font-size: 14px;
        }
        #imageLabel {
            border: 2px dashed #0078D7;
            background-color: #FFFFFF;
            font-size: 14px;
            color: #777;
            padding: 20px;
        }
        #textArea {
            background-color: white;
            border: 1px solid #C0C0C0;
            font-size: 14px;
            padding: 10px;
            border-radius: 5px;
        }
        #uploadButton:hover{
            background-color: #005A9E;
        }
        #uploadButton {
            background-color: #347aa8;
            height: 40px;
        }
        #processButton:hover{
            background-color: #36b08e;
        }
        #processButton {
            background-color: #218d84;
            height: 40px;
        }
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())
