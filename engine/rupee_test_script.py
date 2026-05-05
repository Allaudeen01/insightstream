import matplotlib
matplotlib.use('Agg')
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4

# Register font
font_dir = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(font_dir, 'DejaVuSans.ttf')))

# Create test PDF
output_path = 'C:/Users/ALI/Downloads/rupee_test.pdf'
doc = SimpleDocTemplate(output_path, pagesize=A4)

style = ParagraphStyle('test', fontName='DejaVuSans', fontSize=24)
elements = [
    Paragraph('Test 1 - Direct: ₹2.53 Cr', style),
    Spacer(1, 20),
    Paragraph('<font name="DejaVuSans">Test 2 - Tagged: ₹25.3K</font>',
              ParagraphStyle('t2', fontSize=24)),
    Spacer(1, 20),
    Paragraph('Test 3 - ASCII fallback: Rs. 2.53 Cr', style),
]
doc.build(elements)
print(f'Test PDF created: {output_path}')
print(f'File size: {os.path.getsize(output_path)} bytes')
