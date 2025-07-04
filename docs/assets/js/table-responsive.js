// Table Responsive JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Function to handle table responsiveness
    function handleTableResponsiveness() {
        const tables = document.querySelectorAll('.table-responsive table');
        
        tables.forEach(table => {
            const wrapper = table.closest('.table-responsive');
            
            // Check if table is too wide for container
            function checkTableWidth() {
                const containerWidth = wrapper.offsetWidth;
                const tableWidth = table.scrollWidth;
                
                if (tableWidth > containerWidth) {
                    // Table is too wide, enable horizontal scroll
                    wrapper.style.overflowX = 'auto';
                    table.style.minWidth = '600px';
                } else {
                    // Table fits, disable horizontal scroll
                    wrapper.style.overflowX = 'visible';
                    table.style.minWidth = 'auto';
                }
            }
            
            // Initial check
            checkTableWidth();
            
            // Check on window resize
            window.addEventListener('resize', checkTableWidth);
        });
    }
    
    // Function to add mobile-friendly table features
    function enhanceTableForMobile() {
        const tables = document.querySelectorAll('.table-responsive table');
        
        tables.forEach(table => {
            // Add touch scrolling for mobile
            let isScrolling = false;
            let startX = 0;
            let scrollLeft = 0;
            
            table.addEventListener('touchstart', function(e) {
                isScrolling = true;
                startX = e.touches[0].pageX - table.offsetLeft;
                scrollLeft = table.scrollLeft;
            });
            
            table.addEventListener('touchmove', function(e) {
                if (!isScrolling) return;
                e.preventDefault();
                const x = e.touches[0].pageX - table.offsetLeft;
                const walk = (x - startX) * 2;
                table.scrollLeft = scrollLeft - walk;
            });
            
            table.addEventListener('touchend', function() {
                isScrolling = false;
            });
        });
    }
    
    // Function to add table sorting (optional)
    function addTableSorting() {
        const tables = document.querySelectorAll('.table-responsive table');
        
        tables.forEach(table => {
            const headers = table.querySelectorAll('th');
            
            headers.forEach((header, index) => {
                // Skip if header has no text content
                if (!header.textContent.trim()) return;
                
                // Add sort indicator
                header.style.cursor = 'pointer';
                header.innerHTML += ' <span class="sort-indicator">↕</span>';
                
                header.addEventListener('click', function() {
                    const tbody = table.querySelector('tbody');
                    const rows = Array.from(tbody.querySelectorAll('tr'));
                    const isAscending = header.classList.contains('sort-asc');
                    
                    // Remove sort classes from all headers
                    headers.forEach(h => {
                        h.classList.remove('sort-asc', 'sort-desc');
                        h.querySelector('.sort-indicator').textContent = '↕';
                    });
                    
                    // Sort rows
                    rows.sort((a, b) => {
                        const aValue = a.cells[index].textContent.trim();
                        const bValue = b.cells[index].textContent.trim();
                        
                        // Try to parse as numbers
                        const aNum = parseFloat(aValue);
                        const bNum = parseFloat(bValue);
                        
                        if (!isNaN(aNum) && !isNaN(bNum)) {
                            return isAscending ? bNum - aNum : aNum - bNum;
                        }
                        
                        // Sort as strings
                        return isAscending ? 
                            bValue.localeCompare(aValue) : 
                            aValue.localeCompare(bValue);
                    });
                    
                    // Reorder rows
                    rows.forEach(row => tbody.appendChild(row));
                    
                    // Update header state
                    header.classList.add(isAscending ? 'sort-desc' : 'sort-asc');
                    header.querySelector('.sort-indicator').textContent = 
                        isAscending ? '↓' : '↑';
                });
            });
        });
    }
    
    // Function to add table search functionality
    function addTableSearch() {
        const tables = document.querySelectorAll('.table-responsive');
        
        tables.forEach(wrapper => {
            const table = wrapper.querySelector('table');
            const searchInput = document.createElement('input');
            
            searchInput.type = 'text';
            searchInput.placeholder = 'Search table...';
            searchInput.className = 'table-search';
            searchInput.style.cssText = `
                width: 100%;
                padding: 0.5rem;
                margin-bottom: 1rem;
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                font-size: 0.9rem;
            `;
            
            wrapper.insertBefore(searchInput, table);
            
            searchInput.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                const rows = table.querySelectorAll('tbody tr');
                
                rows.forEach(row => {
                    const text = row.textContent.toLowerCase();
                    row.style.display = text.includes(searchTerm) ? '' : 'none';
                });
            });
        });
    }
    
    // Function to add table export functionality
    function addTableExport() {
        const tables = document.querySelectorAll('.table-responsive');
        
        tables.forEach(wrapper => {
            const table = wrapper.querySelector('table');
            const exportButton = document.createElement('button');
            
            exportButton.textContent = 'Export CSV';
            exportButton.className = 'table-export';
            exportButton.style.cssText = `
                padding: 0.5rem 1rem;
                margin-bottom: 1rem;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
            `;
            
            wrapper.insertBefore(exportButton, table);
            
            exportButton.addEventListener('click', function() {
                const headers = Array.from(table.querySelectorAll('th'))
                    .map(th => th.textContent.trim());
                const rows = Array.from(table.querySelectorAll('tbody tr'))
                    .map(row => 
                        Array.from(row.querySelectorAll('td'))
                            .map(td => td.textContent.trim())
                    );
                
                const csvContent = [
                    headers.join(','),
                    ...rows.map(row => row.join(','))
                ].join('\n');
                
                const blob = new Blob([csvContent], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'table-export.csv';
                a.click();
                window.URL.revokeObjectURL(url);
            });
        });
    }
    
    // Initialize all table features
    handleTableResponsiveness();
    enhanceTableForMobile();
    
    // Optional features (uncomment to enable)
    // addTableSorting();
    // addTableSearch();
    // addTableExport();
    
    // Handle window resize
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            handleTableResponsiveness();
        }, 250);
    });
});

// Add CSS for sort indicators
const style = document.createElement('style');
style.textContent = `
    .sort-indicator {
        font-size: 0.8rem;
        color: #6b7280;
        margin-left: 0.5rem;
    }
    
    th.sort-asc .sort-indicator,
    th.sort-desc .sort-indicator {
        color: #2563eb;
    }
    
    .table-search:focus {
        outline: none;
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    .table-export:hover {
        background: #1d4ed8 !important;
    }
    
    @media (max-width: 768px) {
        .table-search,
        .table-export {
            font-size: 0.8rem;
            padding: 0.4rem;
        }
    }
`;
document.head.appendChild(style); 