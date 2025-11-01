aws ecr create-repository --repository-name smart-cattle-counter
aws ecr get-login-password | docker login --username AWS --password-stdin <your-aws-id>.dkr.ecr.<region>.amazonaws.com
docker build -t smart-cattle-counter .
docker push <your-aws-id>.dkr.ecr.<region>.amazonaws.com/smart-cattle-counter
