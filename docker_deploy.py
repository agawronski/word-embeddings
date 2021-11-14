--------------------------------------------------------------------------------
ssh -i /Users/aidangawronski/Documents/fourth_brain_capstone/fourth-brain-basic-ec2.pem ec2-user@ec2-54-186-106-158.us-west-2.compute.amazonaws.com

sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start

# make an ECR repository and get the URI
414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_jstor_v1

# build the image
sudo docker build -t 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_jstor_v1:latest .

# test that the build runs
docker run --publish 8501:8501 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_jstor_v1:latest

# login
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_jstor_v1

# push the image to ECR
docker push 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_jstor_v1:latest

# Create a cluster 1 time

# create a task definiton based on the image glg_jstor_v1:1

# Create a load balancer glg-application

# Create a security group that allows access on 8501

# Create Listener HTTP:8501 and

# Create target group glg-target-group with target type IP

# Skip register targets

# in the cluster, create a fargate service with the task definition




# PRUNE ALL IMAGES
docker rmi $(docker images -a -q)


--------------------------------------------------------------------------------
