import pygame
import cv2
import pytesseract
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_SPACE
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ancient Text Processor")

# Font setup
font = pygame.font.Font(None, 24)

# Function to open file dialog and select image
def load_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    return file_path

# Function to process the image
def process_image(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, processed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Simple thresholding
    return image, processed

# Function to perform OCR
def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to convert OpenCV image to Pygame format
def cv2_to_pygame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rot90(image)  # Rotate for proper orientation
    image = pygame.surfarray.make_surface(image)
    return pygame.transform.scale(image, (400, 300))  # Resize for display

# Load initial image
image_path = load_image()
if image_path:
    original, processed = process_image(image_path)
    text_result = extract_text(processed)
    orig_surf = cv2_to_pygame(original)
    proc_surf = cv2_to_pygame(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))  # Convert grayscale to RGB for Pygame

running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    # Draw images
    if image_path:
        screen.blit(orig_surf, (50, 100))
        screen.blit(proc_surf, (450, 100))

        # Display extracted text
        lines = text_result.split("\n")
        y_offset = 420
        for line in lines[:5]:  # Limit to 5 lines to avoid overflow
            text_surface = font.render(line, True, BLACK)
            screen.blit(text_surface, (50, y_offset))
            y_offset += 30

        # Labels
        orig_text = font.render("Original Image", True, BLACK)
        proc_text = font.render("Processed Image", True, BLACK)
        screen.blit(orig_text, (50, 70))
        screen.blit(proc_text, (450, 70))

    pygame.display.flip()

pygame.quit()
