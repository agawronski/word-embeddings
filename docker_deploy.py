--------------------------------------------------------------------------------
ssh -i /Users/aidangawronski/Documents/fourth_brain_capstone/fourth-brain-basic-ec2.pem ec2-user@ec2-54-186-106-158.us-west-2.compute.amazonaws.com

sudo service docker start

# make an ECR repository and get the URI
414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1

# build the image
sudo docker build -f DockerfileWiki -t 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1:latest .

# test that the build runs
docker run --publish 8501:8501 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1:latest

# login
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1

# push the image to ECR
docker push 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1:latest

# Console go to ECS
# click task definitions
# fargate, glg-wiki-v1 (Task definition name), ecsExecutionRole (task role),
# memory 10 GB, 2 vCPU
# add container, 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1:latest
# port mapping 8501 tcp
# create


# choose glg-cluster
# create service: fargate, choose correct task defintion, glg-service-wiki-v1 (Service name)
# number of tasks 1,
# choose the VPC, and subnets
# choose a security group, which will allow access from the load balancer security group !!!!!
    # Application load balancer
    # create a new load balancer before completing this (EC2 section of AWS)
    # Load balancer: glg-load-balancer-wiki-v1, same VPC and subnets
    # Listener: HTTP 8501 "Create Target Group"
        # Target group: Target Type: IP addresses glg-target-group-wiki-v1 (Target group name)
        # Protocol HTTP Port 8501 (the open port on the container we are running the app)
        # Health checks: HTTP /healthz ... this is because that is how streamlit allows for a health check, would be different with a different app
        # Specify ports 8501 but don't specify IPs
        # Create
    # Go back to the other tab and select the target group on the load balancer "listner" (click refresh)
    # Create the load balancer
# Go back to the tab where we are creating the service
# choose the new load balancer
# click "add to load balancer"
# Production listener port 80:HTTP
# Target group name: choose the new target group
# Create service

glg-load-balancer-wiki-v1-1343157830.us-west-2.elb.amazonaws.com


### BIO BERT

# make an ECR repository and get the URI
414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_bio_bert_v1

# build the image
sudo docker build -f DockerfileWiki -t 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1:latest .

# test that the build runs
docker run --publish 8501:8501 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1:latest

# login
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1

# push the image to ECR
docker push 414854915400.dkr.ecr.us-west-2.amazonaws.com/glg_wiki_v1:latest


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
