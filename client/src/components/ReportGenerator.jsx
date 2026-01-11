import React, { useState } from 'react';
import { Download, Loader } from 'lucide-react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

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
            const canvas = await html2canvas(input, { scale: 2 });
            const imgData = canvas.toDataURL('image/png');
            const pdf = new jsPDF('p', 'mm', 'a4');

            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = pdf.internal.pageSize.getHeight();

            const imgProperties = pdf.getImageProperties(imgData);
            const imgWidth = pdfWidth;
            const imgHeight = (imgProperties.height * imgWidth) / imgProperties.width;

            // Simplified single page fit or multi-page handling
            // For now, simple single long image scaling or just fitting as much as possible
            if (imgHeight > pdfHeight) {
                // If it's too long, we might just scale it to fit or let it stretch (not ideal)
                // MVP: Just add it
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
