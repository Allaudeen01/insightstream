import type { NextConfig } from "next";

const BACKEND = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [
      {
        // All /api/* requests are forwarded to the Python backend.
        // The frontend never makes a cross-origin call — no CORS, no
        // InPrivate-mode 127.0.0.1 blocking.
        source: "/api/:path*",
        destination: `${BACKEND}/:path*`,
      },
    ];
  },
};

export default nextConfig;
