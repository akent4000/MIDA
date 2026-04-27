#!/usr/bin/env bash
# Regenerate the TypeScript API client from the live FastAPI OpenAPI schema.
#
# Prerequisites:
#   - Node.js 20+ (npm)
#   - Backend running at http://127.0.0.1:8000 (uvicorn backend.app.main:app)
#
# Run from the frontend/ directory:
#   ./scripts/generate-api.sh

set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"
SCHEMA_OUT="openapi.json"
CLIENT_OUT="src/api"

echo "== Fetching OpenAPI schema from $API_URL =="
curl -sf "$API_URL/openapi.json" -o "$SCHEMA_OUT"

echo "== Generating TypeScript client into $CLIENT_OUT =="
npx --yes openapi-typescript-codegen \
  --input "$SCHEMA_OUT" \
  --output "$CLIENT_OUT" \
  --client fetch \
  --useOptions \
  --useUnionTypes

echo "== Done. Generated client lives in $CLIENT_OUT/. =="
