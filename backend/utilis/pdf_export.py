from fpdf import FPDF

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Cattle Count Report", ln=True)
    pdf.cell(200, 10, txt=f"Total Count: {data['count']}", ln=True)
    filename = "report.pdf"
    pdf.output(filename)
    return filename
