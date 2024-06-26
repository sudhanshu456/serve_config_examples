

## Create K8s Cluster

gcloud container clusters create kuberay-gpu-cluster \                                                      
    --num-nodes=1 --min-nodes 0 --max-nodes 1 \
    --zone=asia-south1-c --machine-type e2-standard-4 --spot

## Add GPU node

gcloud container node-pools create gpu-node-pool --accelerator type=nvidia-l4-vws,count=1,gpu-driver-version=default --zone asia-south1-c --cluster kuberay-gpu-cluster --num-nodes 1 --min-nodes 0 --max-nodes 1 --machine-type g2-standard-4 --spot 

## Install Ray Operator
   helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1

## Use RayService CRD
