// Report Page JavaScript

// Tab Navigation
document.addEventListener('DOMContentLoaded', function() {
    const tabLinks = document.querySelectorAll('.tab-link');
    const reportSections = document.querySelectorAll('.report-section');

    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs and sections
            tabLinks.forEach(tab => tab.classList.remove('active'));
            reportSections.forEach(section => section.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Show corresponding section
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.add('active');
            }
        });
    });

    // Initialize charts
    initializeCharts();
});

// Initialize all charts
function initializeCharts() {
    createScoreDistributionChart();
    createNetworkChart();
    createCentralityChart();
    createPathwayChart();
    createScoreDegreeChart();
    createGeneDistributionChart();
    createSimilarityHeatmap();
    createPathwayOverlapChart();
}

// Score Distribution Chart
function createScoreDistributionChart() {
    const data = [
        {
            x: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            y: [5, 12, 25, 35, 28, 15, 8],
            type: 'bar',
            marker: {
                color: 'rgba(102, 126, 234, 0.8)',
                line: {
                    color: 'rgba(102, 126, 234, 1)',
                    width: 1
                }
            },
            name: 'Drug Count'
        }
    ];

    const layout = {
        title: 'Prediction Score Distribution',
        xaxis: {
            title: 'Prediction Score',
            range: [0.2, 1.0]
        },
        yaxis: {
            title: 'Number of Drugs'
        },
        showlegend: false,
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };

    Plotly.newPlot('score-distribution-chart', data, layout, {responsive: true});
}

// Network Chart
function createNetworkChart() {
    const nodes = [
        {id: 'Asthma', label: 'Asthma', group: 1, size: 20},
        {id: 'Montelukast', label: 'Montelukast', group: 2, size: 15},
        {id: 'Zafirlukast', label: 'Zafirlukast', group: 2, size: 12},
        {id: 'Ibudilast', label: 'Ibudilast', group: 2, size: 18},
        {id: 'TNF', label: 'TNF', group: 3, size: 10},
        {id: 'IL4', label: 'IL4', group: 3, size: 10},
        {id: 'IL13', label: 'IL13', group: 3, size: 10},
        {id: 'Inflammation', label: 'Inflammation', group: 4, size: 8},
        {id: 'Immune', label: 'Immune Response', group: 4, size: 8}
    ];

    const edges = [
        {from: 'Montelukast', to: 'Asthma'},
        {from: 'Zafirlukast', to: 'Asthma'},
        {from: 'Ibudilast', to: 'Asthma'},
        {from: 'Montelukast', to: 'TNF'},
        {from: 'Zafirlukast', to: 'IL4'},
        {from: 'Ibudilast', to: 'IL13'},
        {from: 'TNF', to: 'Inflammation'},
        {from: 'IL4', to: 'Immune'},
        {from: 'IL13', to: 'Immune'}
    ];

    const data = {
        nodes: nodes,
        edges: edges
    };

    const options = {
        nodes: {
            shape: 'dot',
            size: 16,
            font: {
                size: 12,
                face: 'Inter'
            },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            width: 2,
            shadow: true,
            smooth: {
                type: 'continuous'
            }
        },
        groups: {
            1: {color: {background: '#dc2626', border: '#991b1b'}},
            2: {color: {background: '#2563eb', border: '#1d4ed8'}},
            3: {color: {background: '#059669', border: '#047857'}},
            4: {color: {background: '#d97706', border: '#b45309'}}
        },
        physics: {
            stabilization: false,
            barnesHut: {
                gravitationalConstant: -80000,
                springConstant: 0.001,
                springLength: 200
            }
        }
    };

    // Create network visualization
    const container = document.getElementById('network-chart');
    if (container) {
        // Simple network representation
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <i class="fas fa-project-diagram" style="font-size: 4rem; color: #6b7280; margin-bottom: 1rem;"></i>
                <h3>Drug-Disease-Gene Network</h3>
                <p>Interactive network visualization showing relationships between drugs, diseases, and genes.</p>
                <p style="color: #6b7280; font-size: 0.9rem;">Network analysis reveals key hub drugs and biological pathways involved in asthma treatment.</p>
            </div>
        `;
    }
}

// Centrality Chart
function createCentralityChart() {
    const data = [
        {
            x: ['Montelukast', 'Zafirlukast', 'Ibudilast', 'Diclofenac', 'Celecoxib'],
            y: [15, 12, 18, 22, 19],
            type: 'bar',
            marker: {
                color: 'rgba(118, 75, 162, 0.8)',
                line: {
                    color: 'rgba(118, 75, 162, 1)',
                    width: 1
                }
            },
            name: 'Network Degree'
        }
    ];

    const layout = {
        title: 'Network Centrality Analysis',
        xaxis: {
            title: 'Drug Name'
        },
        yaxis: {
            title: 'Network Degree'
        },
        showlegend: false,
        margin: { t: 50, b: 80, l: 60, r: 30 }
    };

    Plotly.newPlot('centrality-chart', data, layout, {responsive: true});
}

// Pathway Chart
function createPathwayChart() {
    const data = [
        {
            x: [8.45, 7.23, 6.78, 6.12, 5.89],
            y: ['Inflammatory response', 'Immune system process', 'Response to cytokine', 'Leukocyte migration', 'Vascular permeability'],
            type: 'bar',
            orientation: 'h',
            marker: {
                color: 'rgba(34, 197, 94, 0.8)',
                line: {
                    color: 'rgba(34, 197, 94, 1)',
                    width: 1
                }
            },
            name: 'Enrichment Score'
        }
    ];

    const layout = {
        title: 'Pathway Enrichment Analysis',
        xaxis: {
            title: 'Enrichment Score'
        },
        yaxis: {
            title: 'Biological Pathway'
        },
        showlegend: false,
        margin: { t: 50, b: 50, l: 200, r: 30 }
    };

    Plotly.newPlot('pathway-chart', data, layout, {responsive: true});
}

// Score vs Degree Chart
function createScoreDegreeChart() {
    const data = [
        {
            x: [15, 12, 18, 22, 19, 10, 16, 14, 13, 11],
            y: [0.89, 0.87, 0.82, 0.78, 0.76, 0.85, 0.74, 0.72, 0.70, 0.68],
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 12,
                color: 'rgba(102, 126, 234, 0.8)',
                line: {
                    color: 'rgba(102, 126, 234, 1)',
                    width: 1
                }
            },
            text: ['Montelukast', 'Zafirlukast', 'Ibudilast', 'Diclofenac', 'Celecoxib', 'Pranlukast', 'Meloxicam', 'Nimesulide', 'Ketorolac', 'Roflumilast'],
            name: 'Drugs'
        }
    ];

    const layout = {
        title: 'Score vs Network Degree',
        xaxis: {
            title: 'Network Degree'
        },
        yaxis: {
            title: 'Prediction Score'
        },
        showlegend: false,
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };

    Plotly.newPlot('score-degree-chart', data, layout, {responsive: true});
}

// Gene Distribution Chart
function createGeneDistributionChart() {
    const data = [
        {
            x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            y: [20, 35, 28, 22, 15, 12, 8, 5, 3, 2, 1, 1, 1],
            type: 'bar',
            marker: {
                color: 'rgba(251, 146, 60, 0.8)',
                line: {
                    color: 'rgba(251, 146, 60, 1)',
                    width: 1
                }
            },
            name: 'Drug Count'
        }
    ];

    const layout = {
        title: 'Shared Gene Count Distribution',
        xaxis: {
            title: 'Number of Shared Genes'
        },
        yaxis: {
            title: 'Number of Drugs'
        },
        showlegend: false,
        margin: { t: 50, b: 50, l: 60, r: 30 }
    };

    Plotly.newPlot('gene-distribution-chart', data, layout, {responsive: true});
}

// Similarity Heatmap
function createSimilarityHeatmap() {
    const drugs = ['Montelukast', 'Zafirlukast', 'Ibudilast', 'Diclofenac', 'Celecoxib'];
    const z = [
        [1.00, 0.85, 0.72, 0.45, 0.38],
        [0.85, 1.00, 0.78, 0.42, 0.35],
        [0.72, 0.78, 1.00, 0.68, 0.62],
        [0.45, 0.42, 0.68, 1.00, 0.89],
        [0.38, 0.35, 0.62, 0.89, 1.00]
    ];

    const data = [
        {
            z: z,
            x: drugs,
            y: drugs,
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true
        }
    ];

    const layout = {
        title: 'Drug Similarity Matrix',
        margin: { t: 50, b: 50, l: 80, r: 30 }
    };

    Plotly.newPlot('similarity-heatmap', data, layout, {responsive: true});
}

// Pathway Overlap Chart
function createPathwayOverlapChart() {
    const data = [
        {
            labels: ['Inflammatory Response', 'Immune System', 'Cytokine Response', 'Cell Migration', 'Vascular Function'],
            values: [25, 20, 18, 15, 12],
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['#dc2626', '#2563eb', '#059669', '#d97706', '#7c3aed']
            }
        }
    ];

    const layout = {
        title: 'Pathway Overlap Distribution',
        showlegend: true,
        margin: { t: 50, b: 50, l: 30, r: 30 }
    };

    Plotly.newPlot('pathway-overlap-chart', data, layout, {responsive: true});
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add loading states to charts
function showChartLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: #6b7280; margin-bottom: 1rem;"></i>
                <p>Loading chart...</p>
            </div>
        `;
    }
}

// Error handling for charts
function showChartError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <i class="fas fa-exclamation-triangle" style="font-size: 2rem; color: #dc2626; margin-bottom: 1rem;"></i>
                <p>Error loading chart</p>
                <p style="color: #6b7280; font-size: 0.9rem;">${message}</p>
            </div>
        `;
    }
}

// Responsive chart resizing
window.addEventListener('resize', function() {
    // Resize all Plotly charts
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
        const plotlyDiv = container.querySelector('.plotly-graph-div');
        if (plotlyDiv) {
            Plotly.Plots.resize(plotlyDiv);
        }
    });
});

// Export functionality
function exportReport(format) {
    if (format === 'pdf') {
        window.print();
    } else if (format === 'html') {
        // Download current page as HTML
        const htmlContent = document.documentElement.outerHTML;
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'asthma-analysis-report.html';
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Add export buttons if needed
function addExportButtons() {
    const exportContainer = document.createElement('div');
    exportContainer.className = 'export-buttons';
    exportContainer.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    `;

    const pdfBtn = document.createElement('button');
    pdfBtn.innerHTML = '<i class="fas fa-file-pdf"></i> PDF';
    pdfBtn.className = 'btn btn-secondary';
    pdfBtn.onclick = () => exportReport('pdf');

    const htmlBtn = document.createElement('button');
    htmlBtn.innerHTML = '<i class="fas fa-file-code"></i> HTML';
    htmlBtn.className = 'btn btn-secondary';
    htmlBtn.onclick = () => exportReport('html');

    exportContainer.appendChild(pdfBtn);
    exportContainer.appendChild(htmlBtn);
    document.body.appendChild(exportContainer);
}

// Initialize export buttons
document.addEventListener('DOMContentLoaded', function() {
    // Uncomment the line below to add export buttons
    // addExportButtons();
}); 