    document.addEventListener('DOMContentLoaded', function() {
        const medicineInput = document.getElementById('medicine_name');
        const suggestionsBox = document.getElementById('suggestions');

        medicineInput.addEventListener('input', function() {
            const query = medicineInput.value;
        
            if (query.length > 0) {  // Start searching after at least 2 characters
                fetch(`/autocomplete?query=${query}`)
                    .then(response => {
                        console.log("Autocomplete Response Status:", response.status); // Debugging
                        return response.json();
                    })
                    .then(data => {
                        console.log("Autocomplete Suggestions:", data); // Debugging
                        suggestionsBox.innerHTML = ''; // Clear previous suggestions
                        data.suggestions.forEach(suggestion => {
                            const div = document.createElement('div');
                            div.classList.add('suggestion-item');
                            div.textContent = suggestion;
                            div.addEventListener('click', () => {
                                medicineInput.value = suggestion; // Fill input with selected suggestion
                                suggestionsBox.innerHTML = ''; // Clear suggestions
                            });
                            suggestionsBox.appendChild(div);
                        });
                        suggestionsBox.style.display = 'block'; // Show suggestions box
                    });
            } else {
                suggestionsBox.innerHTML = ''; // Clear suggestions if input length is less than 2
                suggestionsBox.style.display = 'none'; // Hide suggestions box
            }
        });

        // Hide suggestions when clicking outside the input
        document.addEventListener('click', function(e) {
            if (!suggestionsBox.contains(e.target) && e.target !== medicineInput) {
                suggestionsBox.style.display = 'none';
            }
        });
    });
