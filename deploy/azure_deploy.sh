#!/usr/bin/env bash
# =============================================================================
# Deploy the Agentic RAG API to Azure Container Apps — the "proper" path:
# user-assigned managed identity for the image pull (AcrPull) AND for reading
# secrets from Key Vault. No admin passwords, no keys in your shell.
#
# Prereqs: `az` CLI logged in (az account show works), Docker with buildx, and a
#          `.env` in the repo root with the 3 API keys.
# Usage:   ./deploy/azure_deploy.sh
# =============================================================================
set -euo pipefail

# ---- Config (edit or override via env) --------------------------------------
RG="${RG:-rg-demo}"
LOC="${LOC:-northeurope}"
ACR="${ACR:-testisaac}"                 # existing registry; must be globally unique if new
KV="${KV:-agenticragkv0725}"            # Key Vault name; globally unique, 3-24 chars
MI="${MI:-agentic-rag-mi}"              # user-assigned managed identity
ENVN="${ENVN:-agentic-rag-env}"
APP="${APP:-agentic-rag}"
IMG="${IMG:-agentic-rag-api:v1}"
CPU="${CPU:-1.0}"
MEMORY="${MEMORY:-2.0Gi}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ -f .env ] || { echo "ERROR: .env not found in $ROOT"; exit 1; }

# ---- One-time subscription setup --------------------------------------------
echo "==> extension + provider registration"
az extension add --name containerapp --upgrade >/dev/null
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.KeyVault --wait
az provider register --namespace Microsoft.ManagedIdentity --wait

az group create -n "$RG" -l "$LOC" >/dev/null

# ---- A. ACR + build/push -----------------------------------------------------
# We build locally and push (docker buildx). `az acr build` would build in the
# cloud, but ACR Tasks is disabled on many free/sponsored subs (TasksOperationsNotAllowed).
echo "==> [A] ACR + build/push (amd64)"
az acr create -n "$ACR" -g "$RG" --sku Basic >/dev/null 2>&1 || echo "    (ACR already exists)"
az acr login -n "$ACR"
docker buildx build --platform linux/amd64 -t "$ACR.azurecr.io/$IMG" --push .

# ---- Managed identity (the app's "role") ------------------------------------
echo "==> managed identity"
az identity create -n "$MI" -g "$RG" -l "$LOC" >/dev/null 2>&1 || echo "    (identity already exists)"
MI_PRINCIPAL="$(az identity show -n "$MI" -g "$RG" --query principalId -o tsv)"
MI_RESID="$(az identity show -n "$MI" -g "$RG" --query id -o tsv)"

# ---- Key Vault (RBAC) + secrets from .env -----------------------------------
echo "==> Key Vault + secrets"
az keyvault create -n "$KV" -g "$RG" -l "$LOC" --enable-rbac-authorization true >/dev/null 2>&1 || echo "    (KV already exists)"
KV_ID="$(az keyvault show -n "$KV" -g "$RG" --query id -o tsv)"
ACR_ID="$(az acr show -n "$ACR" --query id -o tsv)"
ME="$(az ad signed-in-user show --query id -o tsv)"

# You (Owner) need a DATA-plane role to write secrets — Owner alone can't under RBAC.
az role assignment create --assignee "$ME" --role "Key Vault Secrets Officer" --scope "$KV_ID" >/dev/null 2>&1 || true
echo "    waiting 45s for RBAC propagation before writing secrets..."
sleep 45

# read a value from .env without exporting it into the shell environment
val() { grep "^$1=" .env | cut -d= -f2- | tr -d '"' | tr -d "'"; }
az keyvault secret set --vault-name "$KV" -n groq-api-key   --value "$(val GROQ_API_KEY)"   >/dev/null
az keyvault secret set --vault-name "$KV" -n gemini-api-key --value "$(val GEMINI_API_KEY)" >/dev/null
az keyvault secret set --vault-name "$KV" -n serper-api-key --value "$(val SERPER_API_KEY)" >/dev/null

# grant the identity: read Key Vault secrets + pull from ACR
az role assignment create --assignee "$MI_PRINCIPAL" --role "Key Vault Secrets User" --scope "$KV_ID" >/dev/null 2>&1 || true
az role assignment create --assignee "$MI_PRINCIPAL" --role AcrPull --scope "$ACR_ID" >/dev/null 2>&1 || true

# ---- Environment + app -------------------------------------------------------
echo "==> Container Apps environment"
az containerapp env create -n "$ENVN" -g "$RG" -l "$LOC" >/dev/null

echo "==> Container App (identity + Key Vault-referenced secrets + HTTPS ingress)"
az containerapp create -n "$APP" -g "$RG" \
  --environment "$ENVN" \
  --image "$ACR.azurecr.io/$IMG" \
  --target-port 8000 --ingress external \
  --cpu "$CPU" --memory "$MEMORY" --min-replicas 1 --max-replicas 3 \
  --user-assigned "$MI_RESID" \
  --registry-server "$ACR.azurecr.io" --registry-identity "$MI_RESID" \
  --secrets \
    groq=keyvaultref:https://$KV.vault.azure.net/secrets/groq-api-key,identityref:$MI_RESID \
    gemini=keyvaultref:https://$KV.vault.azure.net/secrets/gemini-api-key,identityref:$MI_RESID \
    serper=keyvaultref:https://$KV.vault.azure.net/secrets/serper-api-key,identityref:$MI_RESID \
  --env-vars \
    GROQ_API_KEY=secretref:groq GEMINI_API_KEY=secretref:gemini SERPER_API_KEY=secretref:serper \
  >/dev/null

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo ""
echo "✅ Deployed to Container Apps. Give it ~1-2 min, then (note HTTPS):"
echo "   curl https://$FQDN/health"
echo "   curl -X POST https://$FQDN/query -H 'Content-Type: application/json' -d '{\"query\":\"What is Agentic RAG?\"}'"
echo ""
echo "Tear down with: ./deploy/azure_teardown.sh"
