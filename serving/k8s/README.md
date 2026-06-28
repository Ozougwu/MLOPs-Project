# Kubernetes deployment (creative extension)

Deploys the `online-shoppers-serving` container to a local Kubernetes cluster
(Docker Desktop) with a load-balanced Service and a CPU-based
HorizontalPodAutoscaler. This is **optional** — the Docker image alone satisfies
serving requirement #4; Kubernetes adds production-style replication, health
gating, and elastic autoscaling.

## Manifests

| File | Resource | Purpose |
|---|---|---|
| `deployment.yaml` | Deployment | 2 replicas of the serving image; liveness (`/health`) + readiness (`/ready`) probes; CPU/memory requests + limits |
| `service.yaml` | Service (NodePort) | Stable in-cluster IP load-balancing across pods; reachable on host port `30080` |
| `hpa.yaml` | HorizontalPodAutoscaler | Scales 2→6 replicas at 60% average CPU |

## Run it

```bash
# 1. Build the image (if not already built) so the cluster can use it locally
docker build -f serving/Dockerfile -t online-shoppers-serving:latest .

# 2. Enable Kubernetes in Docker Desktop (Settings -> Kubernetes -> Enable)

# 3. Deploy
kubectl apply -f serving/k8s/

# 4. Wait for pods to be Ready (readiness probe holds traffic until the model loads)
kubectl wait --for=condition=ready pod -l app=online-shoppers-serving --timeout=120s

# 5. Test live through the Service
curl http://localhost:30080/health
curl -X POST http://localhost:30080/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"PageValues": 25.0, "ProductRelated": 50}]}'

# 6. (For the HPA to report CPU) install metrics-server, then patch for
#    Docker Desktop's self-signed kubelet cert:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

# 7. Inspect
kubectl get deployment,pods,svc,hpa -l app=online-shoppers-serving
kubectl top pods -l app=online-shoppers-serving

# 8. Tear down
kubectl delete -f serving/k8s/
```

## Verified result

A live run produced 2 Ready replicas behind the Service, live `/predict`
responses through NodePort 30080, and the HPA reporting real CPU
(`cpu: 5%/60%`, holding at the 2-replica floor under low load). See
`deployment_evidence.txt` for the captured `kubectl` output.
