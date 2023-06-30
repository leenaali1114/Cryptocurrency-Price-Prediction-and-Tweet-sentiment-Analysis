const searchInput = document.getElementById('searchInput');
const tableRows = document.querySelectorAll('tbody tr');

searchInput.addEventListener('input', function() {
    const searchTerm = searchInput.value.trim().toLowerCase();

    tableRows.forEach(function(row) {
    const nameCell = row.querySelector('td:first-child');
    const name = nameCell.textContent.toLowerCase();

    if (name.includes(searchTerm)) {
        row.style.display = '';
    } else {
        row.style.display = 'none';
    }
    });
});