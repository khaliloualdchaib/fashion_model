#!/bin/bash

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

cd api || exit 1
docker build --tag recommendation:v1 .

kubectl apply -f kubernetes/
cd kubernetes || exit 1
kubectl get deployment metrics-server -n kube-system -o yaml > metrics-server.yaml
kubectl apply -f metrics-server.yaml

cd -