#!/usr/bin/env bash
# =============================================================================
# Deploy the Agentic RAG API to AWS ECS Fargate — mirrors the manual steps A→F.
#
# Prereqs: `aws` CLI logged in (aws sts get-caller-identity works), Docker with
#          buildx, and a `.env` in the repo root holding the 3 API keys.
# Usage:   ./deploy/aws_deploy.sh
# Override any setting inline, e.g.:  AWS_REGION=eu-west-1 ./deploy/aws_deploy.sh
# =============================================================================
set -euo pipefail

# ---- Config (edit or override via env) --------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-agentic-rag-api}"
CLUSTER="${CLUSTER:-agentic-rag}"
TASK_FAMILY="${TASK_FAMILY:-agentic-rag}"
LOG_GROUP="${LOG_GROUP:-/ecs/agentic-rag}"
SECRET_NAME="${SECRET_NAME:-agentic-rag/keys}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
CPU="${CPU:-1024}"
MEMORY="${MEMORY:-3072}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ -f .env ] || { echo "ERROR: .env not found in $ROOT"; exit 1; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"
echo "==> account $ACCOUNT_ID | region $AWS_REGION | $ECR_URI:$IMAGE_TAG"

# ---- A. ECR registry + build/push (amd64) -----------------------------------
echo "==> [A] ECR + build/push"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker buildx build --platform linux/amd64 -t "$ECR_URI:$IMAGE_TAG" --push .

# ---- B. CloudWatch log group ------------------------------------------------
echo "==> [B] log group"
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$AWS_REGION" 2>/dev/null \
  || echo "    (log group already exists)"

# ---- C. Secrets Manager (keys read from .env, never printed) -----------------
echo "==> [C] secret"
python3 - <<'PY' > /tmp/agentic_keys.json
import json, pathlib
env = {}
for line in pathlib.Path(".env").read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
json.dump({k: env[k] for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "SERPER_API_KEY")},
          open("/tmp/agentic_keys.json", "w"))
PY
aws secretsmanager create-secret --name "$SECRET_NAME" --region "$AWS_REGION" \
  --secret-string file:///tmp/agentic_keys.json >/dev/null 2>&1 \
  || aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" --region "$AWS_REGION" \
       --secret-string file:///tmp/agentic_keys.json >/dev/null
rm -f /tmp/agentic_keys.json
SECRET_ARN="$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" --query ARN --output text)"

# ---- D. IAM execution role (pull image, write logs, read secret) ------------
echo "==> [D] IAM execution role"
cat > /tmp/ecs-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file:///tmp/ecs-trust.json >/dev/null 2>&1 || echo "    (role already exists)"
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
cat > /tmp/secrets-policy.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"secretsmanager:GetSecretValue","Resource":"$SECRET_ARN"}]}
EOF
aws iam put-role-policy --role-name ecsTaskExecutionRole \
  --policy-name AgenticRagSecretsRead --policy-document file:///tmp/secrets-policy.json
EXEC_ROLE_ARN="$(aws iam get-role --role-name ecsTaskExecutionRole --query 'Role.Arn' --output text)"

# ---- E. Cluster + task definition -------------------------------------------
echo "==> [E] cluster + task definition"
aws ecs create-cluster --cluster-name "$CLUSTER" --region "$AWS_REGION" >/dev/null
cat > /tmp/taskdef.json <<EOF
{
  "family": "$TASK_FAMILY", "networkMode": "awsvpc", "requiresCompatibilities": ["FARGATE"],
  "cpu": "$CPU", "memory": "$MEMORY",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "$EXEC_ROLE_ARN",
  "containerDefinitions": [{
    "name": "agentic-rag", "image": "$ECR_URI:$IMAGE_TAG", "essential": true,
    "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
    "secrets": [
      { "name": "GROQ_API_KEY",   "valueFrom": "$SECRET_ARN:GROQ_API_KEY::" },
      { "name": "GEMINI_API_KEY", "valueFrom": "$SECRET_ARN:GEMINI_API_KEY::" },
      { "name": "SERPER_API_KEY", "valueFrom": "$SECRET_ARN:SERPER_API_KEY::" }
    ],
    "logConfiguration": { "logDriver": "awslogs", "options": {
      "awslogs-group": "$LOG_GROUP", "awslogs-region": "$AWS_REGION", "awslogs-stream-prefix": "ecs" }}
  }]
}
EOF
aws ecs register-task-definition --cli-input-json file:///tmp/taskdef.json --region "$AWS_REGION" >/dev/null

# ---- F. Networking + run + fetch public IP ----------------------------------
echo "==> [F] networking + run"
VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "$AWS_REGION")"
SUBNETS="$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query 'Subnets[].SubnetId' --output text --region "$AWS_REGION" | tr '\t' ',')"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values=agentic-rag-sg --query 'SecurityGroups[0].GroupId' --output text --region "$AWS_REGION" 2>/dev/null || true)"
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  SG_ID="$(aws ec2 create-security-group --group-name agentic-rag-sg \
    --description 'Agentic RAG inbound 8000' --vpc-id "$VPC_ID" --region "$AWS_REGION" --query GroupId --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "$AWS_REGION" >/dev/null
fi

aws ecs run-task --cluster "$CLUSTER" --launch-type FARGATE --task-definition "$TASK_FAMILY" --count 1 \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region "$AWS_REGION" >/dev/null

TASK_ARN="$(aws ecs list-tasks --cluster "$CLUSTER" --query 'taskArns[0]' --output text --region "$AWS_REGION")"
echo "    waiting for task to reach RUNNING..."
aws ecs wait tasks-running --cluster "$CLUSTER" --tasks "$TASK_ARN" --region "$AWS_REGION"
ENI="$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text --region "$AWS_REGION")"
PUBLIC_IP="$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text --region "$AWS_REGION")"

echo ""
echo "✅ Deployed to ECS Fargate. Give it ~1-2 min to load the model + build the index, then:"
echo "   curl http://$PUBLIC_IP:8000/health"
echo "   curl -X POST http://$PUBLIC_IP:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"What is Agentic RAG?\"}'"
echo ""
echo "Tear down with: ./deploy/aws_teardown.sh"
