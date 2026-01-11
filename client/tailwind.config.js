/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: '#3b82f6', // blue-500
                secondary: '#64748b', // slate-500
                accent: '#8b5cf6', // violet-500
                background: '#f8fafc', // slate-50
                surface: '#ffffff',
            }
        },
    },
    plugins: [],
}
