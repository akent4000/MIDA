import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND_URL = process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";
const WS_URL = process.env.VITE_WS_URL ?? "ws://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: BACKEND_URL, changeOrigin: true },
      "/ws": { target: WS_URL, ws: true, changeOrigin: true },
      "/docs": { target: BACKEND_URL, changeOrigin: true },
      "/openapi.json": { target: BACKEND_URL, changeOrigin: true },
    },
  },
  build: {
    sourcemap: true,
    target: "es2022",
  },
});
