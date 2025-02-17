kubectl delete pods --all --grace-period=0 --force
kubectl delete deployments --all --grace-period=0 --force
kubectl delete services --all --grace-period=0 --force
kubectl delete hpa --all --grace-period=0 --force
cd api || exit 1
cd kubernetes || exit 1
docker rmi -f recommendation:v1

cd -