# syntax=docker/dockerfile:1.7
# Multi-stage frontend image: Vite build → Caddy static server.
#
# Build:
#   docker build -f infra/docker/frontend.Dockerfile -t mida-frontend:latest \\
#                --build-arg VITE_API_BASE_URL=https://api.mida.akent.site .
#
# The API base URL is baked in at build time via Vite's import.meta.env.

# ---------------------------------------------------------------------------
# Stage 1: build the Vite bundle
# ---------------------------------------------------------------------------
FROM node:20-alpine AS builder

ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2: serve the bundle via Caddy
# ---------------------------------------------------------------------------
FROM caddy:2-alpine AS prod

# Tiny Caddyfile — static server with SPA fallback for client-side routing.
RUN cat > /etc/caddy/Caddyfile <<'EOF'
:80 {
    root * /srv
    encode gzip zstd

    # SPA fallback — any unmatched path serves index.html
    try_files {path} /index.html
    file_server

    header {
        # Cache hashed asset bundles for a year (Vite emits content-hashed names)
        ?Cache-Control "public, max-age=31536000, immutable"
        # But never cache the entry HTML
        Cache-Control "no-cache, no-store, must-revalidate"
        defer
    }

    @hashed path *.js *.css *.svg *.woff2 *.png
    header @hashed Cache-Control "public, max-age=31536000, immutable"
}
EOF

COPY --from=builder /app/dist /srv

EXPOSE 80
