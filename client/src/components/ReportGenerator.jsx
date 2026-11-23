import React, { useState } from 'react';
import { Download, Loader } from 'lucide-react';

export default function ReportGenerator({ targetId = 'root', fileName = 'report.pdf' }) {
    const [loading, setLoading] = useState(false);

    const handleExport = async () => {
        const input = document.getElementById(targetId);
        if (!input) {
            alert('Element not found');
            return;
        }

        setLoading(true);
        try {
            // Dynamically import heavy PDF and canvas libraries only when requested
            const [jspdfModule, { toPng }] = await Promise.all([
                import('jspdf'),
                import('html-to-image')
            ]);
            const jsPDF = jspdfModule.jsPDF || jspdfModule.default?.jsPDF || jspdfModule.default;

            const imgData = await toPng(input, { cacheBust: true, pixelRatio: 2 });
            const pdf = new jsPDF('p', 'mm', 'a4');

            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = pdf.internal.pageSize.getHeight();

            const imgProperties = pdf.getImageProperties(imgData);
            const imgWidth = pdfWidth;
            const imgHeight = (imgProperties.height * imgWidth) / imgProperties.width;

            // Simplified single page fit or multi-page handling
            if (imgHeight > pdfHeight) {
                pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
            } else {
                pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
            }

            pdf.save(fileName);
        } catch (error) {
            console.error('Report Generation Error:', error);
            alert('Failed to generate PDF. Make sure all assets are loaded.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <button
            onClick={handleExport}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-white rounded-lg font-medium hover:bg-slate-700 transition-colors disabled:opacity-50"
        >
            {loading ? <Loader size={16} className="animate-spin" /> : <Download size={16} />}
            Export Report
        </button>
    );
}

