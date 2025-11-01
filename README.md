#  Smart Cattle Counter

An AI-powered web app that automatically counts cattle in drone images or videos. Upload your footage, let AI detect animals, manually correct results, and export PDF or CSV reports — all through a private, password-protected dashboard.

---

##  Features
- Upload drone **images (JPG/PNG)** or **videos (MP4/MOV)**  
- Automatic cattle detection with AI (GPU accelerated)  
- **Manual corrections**: Add, remove, or adjust detections  
- **PDF/CSV exports** for final reports  
- **Private login system** (JWT-secured)  
- **Cloud-ready** for AWS or GCP deployment  
- **S3 integration** for secure report storage  

---

##  How It Works
1. Record drone footage over your field.  
2. Upload it via the drag-and-drop dashboard.  
3. The AI model detects and counts cattle.  
4. Adjust any errors manually.  
5. Export your final report as PDF or CSV.

---

##  Technology Stack
**Frontend:** React  
**Backend:** FastAPI (Python)  
**AI Model:** YOLO-based cattle detector (custom fine-tuning supported)  
**Storage:** AWS S3 or compatible bucket  
**Database:** PostgreSQL  
**Deployment:** Docker + GCP/AWS scripts  

---

##  Installation (Local)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm start


```
Cloud Deployment

Scripts for GCP and AWS are included in /deploy/.
You’ll need Docker, and access keys for your cloud provider.
Model Training

You can fine-tune the cattle detection model using your own drone footage:

python train.py --data your_dataset.yaml --weights yolov5s.pt --epochs 50
