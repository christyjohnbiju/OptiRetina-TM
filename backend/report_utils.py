from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import os
import datetime
from PIL import Image

def generate_pdf(user_id, prediction, confidence, original_img, gradcam_img, health_tips, pdf_path):
    """
    Generate a PDF report.
    original_img, gradcam_img: numpy arrays (RGB)
    """
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "OptiRetina - DR Analysis Report")
    
    # Metadata
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 100, f"Patient ID: {user_id}")
    
    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 130, f"Prediction: {prediction}")
    c.drawString(250, height - 130, f"Confidence: {confidence:.2f}")

    # Images
    # Convert numpy to PIL for reportlab
    
    # Helper to draw numpy image
    def draw_numpy_image(img_arr, x, y, w, h):
        pil_img = Image.fromarray(img_arr)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        c.drawImage(ImageReader(img_buffer), x, y, width=w, height=h)

    # Draw Original
    c.drawString(50, height - 160, "Fundus Image:")
    draw_numpy_image(original_img, 50, height - 350, 200, 180)

    # Draw Grad-CAM
    c.drawString(300, height - 160, "Grad-CAM Explanation:")
    draw_numpy_image(gradcam_img, 300, height - 350, 200, 180)

    # Health Tips
    c.setFont("Helvetica-Bold", 14)
    y_pos = height - 380
    c.drawString(50, y_pos, "Medical Recommendations:")
    
    c.setFont("Helvetica", 10)
    y_pos -= 20
    for tip in health_tips:
        c.drawString(50, y_pos, f"- {tip}")
        y_pos -= 15

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, "Disclaimer: AI-generated report. Consult a specialist for final diagnosis.")
    
    c.save()
    return pdf_path
