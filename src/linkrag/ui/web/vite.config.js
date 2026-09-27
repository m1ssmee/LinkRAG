import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The API sits at the root (POST /ask, GET /pairs, ...). `npm run dev` forwards it to uvicorn;
// `npm run build` writes dist/, which linkrag.ui.api serves itself.
const api = ['/state', '/upload', '/build', '/sample', '/reset', '/pairs', '/stats', '/ask', '/graph', '/timeline', '/media']

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: Object.fromEntries(api.map((path) => [path, 'http://127.0.0.1:8000'])) },
})
