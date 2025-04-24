import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from main_image_processing import main_func
from OCR.tamil_ocr import get_tamil
from OCR.sanskrit_ocr import get_sanskrit
from AI_Interpretation.sample import get_response
from detect import predict_language

# Global variable to store input image dimensions
input_image_size = None

def process_image():
    global input_image_size
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    try:
        input_img = Image.open(file_path)
        input_image_size = input_img.size

        input_img_tk = ImageTk.PhotoImage(input_img)
        input_label.config(image=input_img_tk)
        input_label.image = input_img_tk

        output_path = main_func(file_path)
        detect_lang = predict_language(output_path)

        if detect_lang == "Sanskrit":
            data = get_sanskrit(output_path)
        elif detect_lang == "Tamil":
            data = get_tamil(output_path)
        else:
            data = "Unknown Language"

        ai_response = get_response(data, detect_lang)

        lang_label.config(text=f"Language: {detect_lang}")

        extracted_text.config(state=tk.NORMAL)
        extracted_text.delete("1.0", tk.END)
        extracted_text.insert(tk.END, data)
        extracted_text.config(state=tk.DISABLED)

        response_text.config(state=tk.NORMAL)
        response_text.delete("1.0", tk.END)
        response_text.insert(tk.END, ai_response)
        response_text.config(state=tk.DISABLED)

        output_img = Image.open(output_path)
        output_img = output_img.resize(input_image_size, Image.Resampling.LANCZOS)
        output_img_tk = ImageTk.PhotoImage(output_img)
        output_label.config(image=output_img_tk)
        output_label.image = output_img_tk

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def display_step_outputs():
    global input_image_size

    if input_image_size is None:
        messagebox.showerror("Error", "Please select and process an image first.")
        return

    for widget in step_outputs_frame.winfo_children():
        widget.destroy()

    step_image_paths = [
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/1_grayscale.png",
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/2_wavelet_denoised.png",
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/3_clahe_enhanced.png",
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/4_esrgan_enhanced.png",
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/5_background_seperated.png",
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/6_savuola_output.png",
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/Step_outputs/7_unsharp_masking_output.png",
        "C:/Users/Nehal/Desktop/Ancient_text/Image_Pipeline/final_output/output.png"
    ]
    step_titles = [
        "Grayscale Image", "Wavelet Denoised Image", "CLAHE Enhanced Image", "ESRGAN Enhanced Image",
        "Background Separated", "Sauvola Output", "Unsharp Masking Output", "Final Output"
    ]
    
    for title, path in zip(step_titles, step_image_paths):
        tk.Label(step_outputs_frame, text=title, font=("Arial", 12, "bold"), bg="#e6e6e6").pack(pady=5)

        try:
            step_img = Image.open(path)
            step_img = step_img.resize(input_image_size, Image.Resampling.LANCZOS)
            step_img_tk = ImageTk.PhotoImage(step_img)

            step_label = tk.Label(step_outputs_frame, image=step_img_tk, bg="#e6e6e6")
            step_label.image = step_img_tk
            step_label.pack()
        except Exception as e:
            tk.Label(step_outputs_frame, text=f"Error loading {title}: {str(e)}", fg="red", bg="#e6e6e6").pack()

    step_outputs_frame.pack()

# GUI Setup
root = tk.Tk()
root.title("Ancient Text Processing")
root.geometry("900x800")
root.configure(bg="#e6e6e6")

# Create a canvas with a scrollbar
canvas = tk.Canvas(root, bg="#ffffff")
scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = ttk.Frame(canvas, style="TFrame")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((450, 0), window=scrollable_frame, anchor="n")
canvas.configure(yscrollcommand=scrollbar.set)

# Style
style = ttk.Style()
style.configure("TFrame", background="#ffffff")
style.configure("TButton", font=("Arial", 11, "bold"), padding=6)
style.configure("TLabel", font=("Arial", 12), background="#ffffff")

# UI Elements Centered
btn_select = ttk.Button(scrollable_frame, text="Select Image", command=process_image)
btn_select.pack(pady=10, anchor="center") # Align to center

input_label = tk.Label(scrollable_frame, bg="#e6e6e6")
input_label.pack(anchor="center") # Align to center

output_label = tk.Label(scrollable_frame, bg="#e6e6e6")
output_label.pack(anchor="center") # Align to center

lang_label = ttk.Label(scrollable_frame, text="Language: ", wraplength=800, justify="center", font=("Arial", 12, "bold"))
lang_label.pack(pady=5, anchor="center") # Align to center

ttk.Label(scrollable_frame, text="Extracted Text:", font=("Arial", 12, "bold")).pack(anchor="center") # Align to center
extracted_text = tk.Text(scrollable_frame, wrap=tk.WORD, font=("Arial", 10), height=15,width=60 ,bg="white")
extracted_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, anchor="center") # Align to center

ttk.Label(scrollable_frame, text="AI Interpretation:", font=("Arial", 12, "bold")).pack(anchor="center") # Align to center
response_text = tk.Text(scrollable_frame, wrap=tk.WORD, font=("Arial", 10), height=15,width=60, bg="white")
response_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, anchor="center") # Align to center

btn_step_outputs = ttk.Button(scrollable_frame, text="Step Outputs", command=display_step_outputs)
btn_step_outputs.pack(pady=10, anchor="center") # Align to center

# Step Outputs Frame (hidden until button click)
step_outputs_frame = tk.Frame(scrollable_frame, bg="#e6e6e6")

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

root.mainloop()