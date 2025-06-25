// Dashboard charts using the pandemic data API

// Function to fetch data from API
async function fetchTimeSeriesData(state, metric) {
    try {
        const response = await fetch(`/stats/api/time-series/${state}/${metric}`);
        return await response.json();
    } catch (error) {
        console.error('Error fetching data:', error);
        return { dates: [], values: [] };
    }
}

// Function to initialize all charts
async function initCharts() {
    // Default state
    const defaultState = 'NY';
    
    // Initialize cases chart
    const casesData = await fetchTimeSeriesData(defaultState, 'positive');
    createLineChart('casesChart', 'COVID-19 Cases', casesData.dates, casesData.values);
    
    // Initialize deaths chart
    const deathsData = await fetchTimeSeriesData(defaultState, 'death');
    createLineChart('deathsChart', 'COVID-19 Deaths', deathsData.dates, deathsData.values);
    
    // Initialize testing chart
    const testingData = await fetchTimeSeriesData(defaultState, 'totalTestResults');
    createLineChart('testingChart', 'COVID-19 Tests', testingData.dates, testingData.values);
    
    // Initialize vaccination chart
    const vaxData = await fetchTimeSeriesData(defaultState, 'Complete_Total_pct');
    createLineChart('vaccinationChart', 'Vaccination Rate (%)', vaxData.dates, vaxData.values);
    
    // Set up state selector
    setupStateSelector();
}

// Function to create a line chart
function createLineChart(elementId, title, labels, data) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: title,
                data: data,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                pointRadius: 0,
                pointHitRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month'
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: title
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: title
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Function to set up state selector
async function setupStateSelector() {
    try {
        // Fetch list of states
        const response = await fetch('/stats/api/states');
        const states = await response.json();
        
        // Get the select element
        const stateSelect = document.getElementById('stateSelector');
        
        // Add options
        states.forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            stateSelect.appendChild(option);
        });
        
        // Set default value
        stateSelect.value = 'NY';
        
        // Add event listener
        stateSelect.addEventListener('change', async (event) => {
            const selectedState = event.target.value;
            updateCharts(selectedState);
        });
    } catch (error) {
        console.error('Error setting up state selector:', error);
    }
}

// Function to update all charts with new state data
async function updateCharts(state) {
    // Update cases chart
    const casesData = await fetchTimeSeriesData(state, 'positive');
    updateChart('casesChart', casesData.dates, casesData.values);
    
    // Update deaths chart
    const deathsData = await fetchTimeSeriesData(state, 'death');
    updateChart('deathsChart', deathsData.dates, deathsData.values);
    
    // Update testing chart
    const testingData = await fetchTimeSeriesData(state, 'totalTestResults');
    updateChart('testingChart', testingData.dates, testingData.values);
    
    // Update vaccination chart
    const vaxData = await fetchTimeSeriesData(state, 'Complete_Total_pct');
    updateChart('vaccinationChart', vaxData.dates, vaxData.values);
}

// Function to update a specific chart
function updateChart(chartId, labels, data) {
    const chart = Chart.getChart(chartId);
    
    if (chart) {
        chart.data.labels = labels;
        chart.data.datasets[0].data = data;
        chart.update();
    }
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', initCharts);