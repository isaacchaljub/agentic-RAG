# Deploy scripts

One-shot scripts that reproduce the manual CLI deploys of the Agentic RAG API
(the FastAPI service, `serving_api/main_v2.py`) to AWS and Azure.

| Script | What it does |
|---|---|
| `aws_deploy.sh` | ECR → build/push (amd64) → CloudWatch log group → Secrets Manager → IAM execution role → ECS cluster + task definition → VPC/SG → `run-task` → prints the public URL |
| `aws_teardown.sh` | Stops the task and deletes SG, log group, secret, ECR repo, cluster |
| `azure_deploy.sh` | Resource group → ACR → build/push (amd64) → managed identity → Key Vault + secrets → role grants (AcrPull, Key Vault Secrets User) → Container Apps env + app → prints the HTTPS URL |
| `azure_teardown.sh` | Deletes the whole resource group |

## Prerequisites
- **AWS:** `aws` CLI authenticated (`aws sts get-caller-identity` works)
- **Azure:** `az` CLI authenticated (`az account show` works)
- **Both:** Docker with `buildx`, and a **`.env`** in the repo root containing:
  ```env
  GROQ_API_KEY=...
  GEMINI_API_KEY=...
  SERPER_API_KEY=...
  ```
  Secrets are read from `.env` at deploy time and pushed into Secrets Manager /
  Key Vault — they are **never** baked into the image or printed.

## Usage
```bash
./deploy/aws_deploy.sh          # or:  AWS_REGION=eu-west-1 ./deploy/aws_deploy.sh
./deploy/aws_teardown.sh

./deploy/azure_deploy.sh        # defaults to RG=rg-demo, ACR=testisaac
./deploy/azure_teardown.sh
```
Every setting is an overridable env var at the top of each script (region, names,
CPU/memory). Create commands are written to be re-runnable (they skip resources
that already exist).

## Notes
- **Architecture:** images are built `--platform linux/amd64` (Macs are arm64;
  Fargate & Container Apps default to x86_64).
- **Azure build:** uses local `docker buildx --push` rather than `az acr build`,
  because ACR Tasks (server-side build) is disabled on many free/sponsored subs.
- **Azure teardown** deletes the entire resource group — safe only because
  `rg-demo` is dedicated to this app.
