from fastapi import FastAPI, UploadFile, Form, Depends
from auth import get_current_user
from detection import detect_cattle
from utils.pdf_export import generate_pdf
from utils.s3_upload import upload_to_s3

app = FastAPI(title="Smart Cattle Counter")

@app.post("/upload")
async def upload_file(file: UploadFile, user=Depends(get_current_user)):
    results = await detect_cattle(file)
    return {"status": "processed", "count": len(results), "detections": results}

@app.post("/export/pdf")
async def export_pdf(data: dict, user=Depends(get_current_user)):
    pdf_path = generate_pdf(data)
    url = upload_to_s3(pdf_path)
    return {"message": "Report generated", "pdf_url": url}
