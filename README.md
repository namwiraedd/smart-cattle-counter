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


Quick Setup checklist (TF OD API essentials)

You must do these before running scripts:

Install TensorFlow 2.x (GPU recommended):

pip install tensorflow==2.11.0   # choose the TF version that matches your CUDA


Install Object Detection API (official steps):

Clone models repo:

git clone https://github.com/tensorflow/models.git
cd models/research


Install protos & TFOD dependencies (follow TF repo docs). Typical steps:

# from models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
pip install -r object_detection/packages/tf2/requirements.txt


Ensure PYTHONPATH includes models/research and models/research/slim:

export PYTHONPATH=$PYTHONPATH:/path/to/models/research:/path/to/models/research/slim


Install COCO tools:

pip install pycocotools


Ensure yolo_dataset exists and has images/labels as expected.

Choose a base pretrained checkpoint and set --pretrained_ckpt when running train_model.py. Common choices are TF2 detection model checkpoints available from TF model zoo.

Example run (after TF OD API setup)

Convert and train:

# from repo root
python backend/train_model.py \
  --yolo_ds ./yolo_dataset \
  --output_dir ./tf_od/data \
  --model_dir ./tf_od/models/exp1 \
  --num_steps 20000 \
  --batch_size 4 \
  --pretrained_ckpt /path/to/pretrained/checkpoint/ckpt-0


Evaluate:

python backend/evaluate_model.py \
  --pipeline_config ./tf_od/data/pipeline.config \
  --model_dir ./tf_od/models/exp1 \
  --out_dir ./tf_od/eval_reports


Monitor training with TensorBoard:

tensorboard --logdir ./tf_od/models/exp1

