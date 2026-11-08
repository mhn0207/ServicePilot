import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/openapi.json': {
        target: process.env.VITE_PYTHON_API_URL || 'http://localhost:8000',
        changeOrigin: true
      },
      '/api/zhiying': {
        target: process.env.VITE_PYTHON_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/zhiying/, '')
      }
    }
  }
})
