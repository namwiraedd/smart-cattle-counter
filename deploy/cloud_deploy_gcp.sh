gcloud builds submit --tag gcr.io/your-project-id/smart-cattle-counter
gcloud run deploy smart-cattle-counter --image gcr.io/your-project-id/smart-cattle-counter --platform managed
