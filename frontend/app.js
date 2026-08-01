const API_BASE_URL = window.location.origin;
let metricsChart;
let selectedMetric = 'Accuracy';
let metricOptions = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'];

const statusBox = document.getElementById('statusBox');
const summaryCards = document.getElementById('summaryCards');
const chartTypeButtons = document.getElementById('chartTypeButtons');
const metricButtons = document.getElementById('metricButtons');
const bestModelCard = document.getElementById('bestModelCard');
const topModelsBody = document.getElementById('topModelsBody');
const modelDetails = document.getElementById('modelDetails');
const comparisonChartCanvas = document.getElementById('metricsChart');
const scoreChartCanvas = document.getElementById('scoreChart');
const importanceChartCanvas = document.getElementById('importanceChart');
const importanceSummary = document.getElementById('importanceSummary');
const importanceModelSelect = document.getElementById('importanceModelSelect');
const leaderboardMetric1Header = document.getElementById('leaderboardMetric1Header');
const leaderboardMetric2Header = document.getElementById('leaderboardMetric2Header');
const downloadsSection = document.getElementById('downloadsSection');
const downloadButtons = document.getElementById('downloadButtons');
const predictionSection = document.getElementById('predictionSection');
const predictionFeatureGrid = document.getElementById('predictionFeatureGrid');
const predictionResultBox = document.getElementById('predictionResultBox');
const predictButton = document.getElementById('predictButton');
const toastContainer = document.getElementById('toastContainer');
const datasetFileInput = document.getElementById('datasetFile');
const targetColumnInput = document.getElementById('targetColumn');
const runButton = document.getElementById('runButton');

let pipelineStatusTimer = null;
window.currentJobId = null;
let selectedChartType = 'bar';
let scoreDistributionChart;
let importanceFeatureChart;
let selectedImportanceModel = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatApiError(payload, fallbackMessage) {
  if (!payload) {
    return fallbackMessage;
  }

  if (typeof payload.detail === 'string') {
    return payload.detail;
  }

  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((item) => item.msg || item.message || JSON.stringify(item))
      .join('\n');
  }

  if (typeof payload.error === 'string') {
    return payload.error;
  }

  return fallbackMessage;
}

function showToast(message, variant = 'success') {
  if (!toastContainer || !window.bootstrap) {
    return;
  }

  const toastElement = document.createElement('div');
  toastElement.className = `toast align-items-center text-bg-${variant} border-0 notification-toast`;
  toastElement.role = 'alert';
  toastElement.ariaLive = 'assertive';
  toastElement.ariaAtomic = 'true';
  toastElement.innerHTML = `
    <div class="d-flex">
      <div class="toast-body"></div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
  `;
  toastElement.querySelector('.toast-body').textContent = message;
  toastContainer.appendChild(toastElement);

  const toast = bootstrap.Toast.getOrCreateInstance(toastElement, { delay: 4500 });
  toast.show();
  toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
}

function getMetricOptions(result) {
  if (Array.isArray(result.metric_options) && result.metric_options.length) {
    return result.metric_options;
  }

  return result.problem_type === 'regression'
    ? ['R2', 'RMSE', 'MAE', 'MSE']
    : ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'];
}

function getLeaderboardMetrics(result) {
  if (Array.isArray(result.leaderboard_metrics) && result.leaderboard_metrics.length) {
    return result.leaderboard_metrics;
  }

  return result.problem_type === 'regression'
    ? ['R2', 'RMSE']
    : ['Accuracy', 'F1 Score'];
}

function isClassification(result) {
  return result.problem_type !== 'regression';
}

function clearPipelineStatusTimer() {
  if (pipelineStatusTimer) {
    clearInterval(pipelineStatusTimer);
    pipelineStatusTimer = null;
  }
}

function setStatus(message, type = 'info', progress = null) {
  statusBox.classList.remove('d-none');
  const alertType = type === 'error' ? 'danger' : type === 'warning' ? 'warning' : type === 'success' ? 'success' : 'light';
  statusBox.className = `alert alert-${alertType} status-box mb-0`;
  if (progress === null) {
    statusBox.innerHTML = `<div>${escapeHtml(message).replaceAll('\n', '<br />')}</div>`;
    return;
  }

  const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
  const animatedClass = safeProgress < 100 && type !== 'error' ? 'progress-bar-striped progress-bar-animated' : '';
  statusBox.innerHTML = `
    <div class="d-flex justify-content-between align-items-center gap-3 mb-2 flex-wrap">
      <div>${escapeHtml(message)}</div>
      <div class="fw-semibold">${safeProgress}%</div>
    </div>
    <div class="progress progress-thin">
      <div class="progress-bar ${animatedClass}" role="progressbar" style="width: ${safeProgress}%" aria-valuenow="${safeProgress}" aria-valuemin="0" aria-valuemax="100"></div>
    </div>
  `;
}

function setRunningState(isRunning) {
  runButton.disabled = isRunning;
  datasetFileInput.disabled = isRunning;
  targetColumnInput.disabled = isRunning;
  runButton.innerHTML = isRunning
    ? '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Training in progress...'
    : 'Run AutoML Pipeline';
}

function setPredictionState(isRunning) {
  if (predictButton) {
    predictButton.disabled = isRunning;
    predictButton.innerHTML = isRunning
      ? '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Predicting...'
      : 'Predict';
  }

  predictionFeatureGrid?.querySelectorAll('input').forEach((input) => {
    input.disabled = isRunning;
  });
}

async function fetchPipelineStatus(jobId) {
  const response = await fetch(`${API_BASE_URL}/pipeline-status/${jobId}`);
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(formatApiError(payload, 'Unable to read pipeline status.'));
  }

  return payload;
}

async function pollPipelineStatus(jobId) {
  clearPipelineStatusTimer();
  let shouldContinuePolling = true;

  const update = async () => {
    try {
      const payload = await fetchPipelineStatus(jobId);
      setStatus(payload.message || 'Training pipeline running...', payload.status === 'failed' ? 'error' : 'info', payload.progress ?? 0);

      if (payload.status === 'completed') {
        clearPipelineStatusTimer();
        setRunningState(false);
        shouldContinuePolling = false;
        renderResult(payload.result);
      }

      if (payload.status === 'failed') {
        clearPipelineStatusTimer();
        setRunningState(false);
        shouldContinuePolling = false;
      }
    } catch (error) {
      if (error.message === 'Pipeline job not found.') {
        clearPipelineStatusTimer();
        setRunningState(false);
        shouldContinuePolling = false;
        setStatus('Pipeline job expired or the server reloaded. Run the pipeline again.', 'error');
        showToast('Pipeline job expired or the server reloaded.', 'danger');
        return;
      }

      clearPipelineStatusTimer();
      setRunningState(false);
      shouldContinuePolling = false;
      setStatus(error.message || 'Unable to track pipeline progress.', 'error');
      showToast(error.message || 'Unable to track pipeline progress.', 'danger');
    }
  };

  await update();

  if (shouldContinuePolling && !pipelineStatusTimer) {
    pipelineStatusTimer = setInterval(update, 900);
  }
}

function formatMetric(value, result, metricName) {
  const numericValue = Number(value);

  if (Number.isNaN(numericValue)) {
    return '—';
  }

  if (isClassification(result)) {
    return `${(numericValue * 100).toFixed(1)}%`;
  }

  if (metricName === 'R2') {
    return numericValue.toFixed(3);
  }

  return numericValue.toFixed(3);
}

function chartTickLabel(value, result) {
  if (isClassification(result)) {
    return `${Number(value).toFixed(1)}%`;
  }

  return Number(value).toFixed(3);
}

function chartLegendLabel(label, value, result) {
  return `${label} (${chartTickLabel(value, result)})`;
}

function buildSummaryCards(result) {
  const dashboardSummary = result.dashboard_summary || {};
  const bestMetricLabel = dashboardSummary.best_metric_label || result.primary_metric || 'Accuracy';
  const bestMetricValue = dashboardSummary.best_metric_value ?? result.best_model?.[bestMetricLabel];
  const metricDisplay = formatMetric(bestMetricValue, result, bestMetricLabel);

  summaryCards.innerHTML = `
    <div class="col-sm-6 col-lg-3">
      <div class="stat-box"><div class="small text-muted">Problem Type</div><div class="value value-sm">${escapeHtml(dashboardSummary.problem_type || result.problem_type || '—')}</div></div>
    </div>
    <div class="col-sm-6 col-lg-3">
      <div class="stat-box"><div class="small text-muted">Rows</div><div class="value">${dashboardSummary.rows ?? result.dataset.rows}</div></div>
    </div>
    <div class="col-sm-6 col-lg-3">
      <div class="stat-box"><div class="small text-muted">Columns</div><div class="value">${dashboardSummary.columns ?? result.dataset.columns}</div></div>
    </div>
    <div class="col-sm-6 col-lg-3">
      <div class="stat-box"><div class="small text-muted">Missing Values</div><div class="value">${dashboardSummary.missing_values ?? result.dataset.missing_values_total ?? 0}</div></div>
    </div>
    <div class="col-sm-6 col-lg-3">
      <div class="stat-box"><div class="small text-muted">Best Model</div><div class="value value-sm">${escapeHtml(dashboardSummary.best_model || result.best_model_name || result.best_model?.Model || '—')}</div></div>
    </div>
    <div class="col-sm-6 col-lg-3">
      <div class="stat-box"><div class="small text-muted">${escapeHtml(bestMetricLabel)}</div><div class="value value-sm">${metricDisplay}</div></div>
    </div>
  `;
}

function renderChartTypeButtons() {
  if (!chartTypeButtons) {
    return;
  }

  const chartTypes = [
    { key: 'bar', label: 'Bar' },
    { key: 'line', label: 'Line' },
    { key: 'radar', label: 'Radar' },
    { key: 'doughnut', label: 'Doughnut' },
  ];

  chartTypeButtons.innerHTML = chartTypes
    .map((type) => `<button class="metric-pill ${type.key === selectedChartType ? 'active' : ''}" type="button" data-chart-type="${type.key}">${type.label}</button>`)
    .join('');

  chartTypeButtons.querySelectorAll('[data-chart-type]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedChartType = button.dataset.chartType;
      renderChartTypeButtons();
      updateComparisonChart();
    });
  });
}

function updateChart() {
  updateComparisonChart();
}

function renderMetricButtons(result = window.currentResult) {
  if (!result) {
    return;
  }

  metricOptions = getMetricOptions(result);
  if (!metricOptions.includes(selectedMetric)) {
    selectedMetric = result.primary_metric || metricOptions[0];
  }

  metricButtons.innerHTML = metricOptions
    .map((metric) => `
      <button class="metric-pill ${metric === selectedMetric ? 'active' : ''}" type="button" data-metric="${metric}">${metric}</button>
    `)
    .join('');

  metricButtons.querySelectorAll('[data-metric]').forEach((button) => {
    button.addEventListener('click', () => {
      selectedMetric = button.dataset.metric;
      renderMetricButtons(window.currentResult);
      updateChart();
    });
  });
}

function getComparisonChartSeries(result) {
  const classificationMode = isClassification(result);
  const sortedComparison = [...result.comparison].sort((left, right) => Number(right[selectedMetric]) - Number(left[selectedMetric]));
  const labels = sortedComparison.map((row) => row.Model);
  const values = sortedComparison.map((row) => (
    classificationMode ? Number(row[selectedMetric]) * 100 : Number(row[selectedMetric])
  ));

  return { labels, values, classificationMode };
}

function getMetricAxisBounds(values) {
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const spread = Math.max(maxValue - minValue, 1);
  const padding = Math.max(spread * 0.12, 0.25);

  return {
    min: Math.max(0, minValue - padding),
    max: maxValue + padding,
  };
}

function updateComparisonChart() {
  if (!window.currentResult || !window.currentResult.comparison) return;

  const result = window.currentResult;
  const { labels, values, classificationMode } = getComparisonChartSeries(result);
  const axisBounds = getMetricAxisBounds(values);

  if (metricsChart) {
    metricsChart.destroy();
  }

  if (!comparisonChartCanvas) return;

  const commonDataset = {
    label: selectedMetric,
    data: values,
    backgroundColor: ['#5dd7ff', '#7c4dff', '#6ee7b7', '#f59e0b', '#fb7185', '#a78bfa'],
    borderColor: '#5dd7ff',
    borderWidth: selectedChartType === 'line' ? 3 : 0,
    fill: selectedChartType === 'line',
    tension: 0.35,
    borderRadius: 10,
    barThickness: selectedChartType === 'bar' ? 24 : undefined,
    maxBarThickness: selectedChartType === 'bar' ? 28 : undefined,
    categoryPercentage: selectedChartType === 'bar' ? 0.72 : undefined,
    barPercentage: selectedChartType === 'bar' ? 0.9 : undefined,
  };

  const chartConfig = {
    type: selectedChartType,
    data: {
      labels,
      datasets: [commonDataset],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      indexAxis: selectedChartType === 'bar' || selectedChartType === 'line' ? 'x' : undefined,
      plugins: {
        legend: {
          display: selectedChartType === 'doughnut' || selectedChartType === 'radar',
          position: 'bottom',
          labels: {
            color: '#c8d7ef',
            usePointStyle: true,
            padding: 16,
            generateLabels(chart) {
              const data = chart.data;
              return data.labels.map((label, index) => ({
                text: chartLegendLabel(label, data.datasets[0].data[index], result),
                fillStyle: data.datasets[0].backgroundColor[index % data.datasets[0].backgroundColor.length],
                strokeStyle: data.datasets[0].backgroundColor[index % data.datasets[0].backgroundColor.length],
                pointStyle: 'circle',
                hidden: false,
                index,
              }));
            },
          },
        },
        tooltip: {
          callbacks: {
            label(context) {
              return chartLegendLabel(context.label, context.raw, result);
            },
          },
        },
      },
      scales: selectedChartType === 'radar'
        ? {
            r: {
              angleLines: { color: 'rgba(255,255,255,0.08)' },
              grid: { color: 'rgba(255,255,255,0.08)' },
              pointLabels: { color: '#c8d7ef' },
              ticks: {
                backdropColor: 'transparent',
                color: '#c8d7ef',
              },
            },
          }
        : selectedChartType === 'doughnut'
          ? {}
          : {
              y: {
                beginAtZero: false,
                min: axisBounds.min,
                max: axisBounds.max,
                ticks: {
                  color: '#c8d7ef',
                  callback: (value) => chartTickLabel(value, result),
                },
                grid: {
                  color: 'rgba(255,255,255,0.08)',
                },
              },
              x: {
                ticks: {
                  color: '#c8d7ef',
                },
                grid: {
                  display: false,
                },
              },
            },
    },
  };

  metricsChart = new Chart(comparisonChartCanvas, chartConfig);
}

function updateScoreDistributionChart(result) {
  if (!scoreChartCanvas || !window.currentResult?.comparison?.length) {
    return;
  }

  if (scoreDistributionChart) {
    scoreDistributionChart.destroy();
  }

  const leaderboardMetrics = getLeaderboardMetrics(result);
  const metricName = leaderboardMetrics[0] || result.primary_metric || selectedMetric;
  const topRows = result.comparison.slice(0, 5);
  const labels = topRows.map((row) => row.Model);
  const values = topRows.map((row) => {
    const value = Number(row[metricName]);
    return isClassification(result) ? value * 100 : value;
  });

  if (!labels.length) {
    return;
  }

  scoreDistributionChart = new Chart(scoreChartCanvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: ['#5dd7ff', '#7c4dff', '#6ee7b7', '#f59e0b', '#fb7185'],
        borderColor: '#07111f',
        borderWidth: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      cutout: '58%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#c8d7ef',
            usePointStyle: true,
            boxWidth: 10,
            padding: 16,
            generateLabels(chart) {
              const data = chart.data;
              return data.labels.map((label, index) => ({
                text: chartLegendLabel(label, data.datasets[0].data[index], result),
                fillStyle: data.datasets[0].backgroundColor[index % data.datasets[0].backgroundColor.length],
                strokeStyle: data.datasets[0].backgroundColor[index % data.datasets[0].backgroundColor.length],
                pointStyle: 'circle',
                hidden: false,
                index,
              }));
            },
          },
        },
        tooltip: {
          callbacks: {
            label(context) {
              return chartLegendLabel(context.label, context.raw, result);
            },
          },
        },
      },
    },
  });
}

function updateFeatureImportanceChart(result) {
  if (!importanceChartCanvas) {
    return;
  }

  const featureImportance = result.feature_importance || {};
  const modelNames = Object.keys(featureImportance);

  if (!modelNames.length) {
    importanceSummary.textContent = 'No explainability data is available for this run.';
    if (importanceModelSelect) {
      importanceModelSelect.innerHTML = '';
    }
    if (importanceFeatureChart) {
      importanceFeatureChart.destroy();
      importanceFeatureChart = undefined;
    }
    return;
  }

  if (importanceModelSelect) {
    const preferredDefault = result.best_model_name || result.best_model?.Model || modelNames[0];
    const selectedModel = modelNames.includes(selectedImportanceModel)
      ? selectedImportanceModel
      : modelNames.includes(preferredDefault)
        ? preferredDefault
        : modelNames[0];
    selectedImportanceModel = selectedModel;
    importanceModelSelect.innerHTML = modelNames
      .map((modelName) => `<option value="${escapeHtml(modelName)}">${escapeHtml(modelName)}</option>`)
      .join('');
    importanceModelSelect.value = selectedImportanceModel;
    importanceModelSelect.onchange = (event) => {
      selectedImportanceModel = event.target.value;
      updateFeatureImportanceChart(window.currentResult);
    };
  }

  const modelName = selectedImportanceModel || modelNames[0];
  const importanceRows = featureImportance[modelName] || [];

  if (!importanceRows.length) {
    importanceSummary.textContent = `No explainability data is available for ${modelName}.`;
    if (importanceFeatureChart) {
      importanceFeatureChart.destroy();
      importanceFeatureChart = undefined;
    }
    return;
  }

  importanceSummary.textContent = `Showing feature importance for ${modelName}.`;

  const labels = importanceRows.slice(0, 8).map((item) => item.feature);
  const values = importanceRows.slice(0, 8).map((item) => Number(item.importance) * 100);

  if (importanceFeatureChart) {
    importanceFeatureChart.destroy();
  }

  importanceFeatureChart = new Chart(importanceChartCanvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Feature Importance',
        data: values,
        backgroundColor: '#5dd7ff',
        borderRadius: 10,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: {
            color: '#c8d7ef',
            callback: (value) => `${Number(value).toFixed(1)}%`,
          },
          grid: { color: 'rgba(255,255,255,0.08)' },
        },
        y: {
          ticks: { color: '#c8d7ef' },
          grid: { display: false },
        },
      },
    },
  });
}

function renderBestModel(result) {
  const model = result.best_model;
  const leaderboardMetrics = getLeaderboardMetrics(result);
  if (!model) {
    bestModelCard.innerHTML = '<p class="text-muted">No best model available.</p>';
    return;
  }

  const metricsMarkup = leaderboardMetrics
    .filter((metric) => Object.prototype.hasOwnProperty.call(model, metric))
    .map((metric) => `
        <div class="d-flex justify-content-between"><span>${metric}</span><strong>${formatMetric(model[metric], result, metric)}</strong></div>
      `)
    .join('');

  bestModelCard.innerHTML = `
    <div class="dark-result-card">
      <h4 class="fw-bold mb-3">${model.Model}</h4>
      <div class="d-grid gap-2">
        ${metricsMarkup}
      </div>
    </div>
  `;
}

function updateLeaderboardHeaders(result) {
  const leaderboardMetrics = getLeaderboardMetrics(result);
  if (leaderboardMetric1Header) {
    leaderboardMetric1Header.textContent = leaderboardMetrics[0] || 'Metric';
  }
  if (leaderboardMetric2Header) {
    leaderboardMetric2Header.textContent = leaderboardMetrics[1] || 'Metric';
  }
}

function renderTopModels(result) {
  const leaderboardMetrics = getLeaderboardMetrics(result);
  updateLeaderboardHeaders(result);
  topModelsBody.innerHTML = result.top3
    .map((row) => `
      <tr>
        <td>${row.Model}</td>
        <td>${formatMetric(row[leaderboardMetrics[0]], result, leaderboardMetrics[0])}</td>
        <td>${formatMetric(row[leaderboardMetrics[1]], result, leaderboardMetrics[1])}</td>
      </tr>
    `)
    .join('');
}

function renderModelDetails(result) {
  const metricNames = getMetricOptions(result);
  const entries = Object.entries(result.evaluation);
  const importanceByModel = result.feature_importance || {};
  modelDetails.innerHTML = entries
    .map(([name, metrics]) => `
      <div class="border rounded-3 p-3 mb-3">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <h4 class="h6 mb-0">${name}</h4>
          <span class="badge bg-secondary">${result.primary_metric || metricNames[0]} ${formatMetric(metrics[result.primary_metric || metricNames[0]], result, result.primary_metric || metricNames[0])}</span>
        </div>
        <div class="row g-2">
          ${metricNames
            .filter((metric) => Object.prototype.hasOwnProperty.call(metrics, metric))
            .map((metric) => `
              <div class="col-sm-6">
                <div class="small text-muted">${metric}</div>
                <div>${formatMetric(metrics[metric], result, metric)}</div>
              </div>
            `)
            .join('')}
        </div>
        ${Array.isArray(importanceByModel[name]) && importanceByModel[name].length ? `
          <div class="mt-3">
            <div class="small text-muted mb-2">Feature Importance</div>
            <div class="feature-importance-list">
              ${importanceByModel[name]
                .map((item) => `
                  <div class="feature-importance-row">
                    <div class="feature-importance-label">${escapeHtml(item.feature)}</div>
                    <div class="feature-importance-bar-wrap">
                      <div class="feature-importance-bar" style="width: ${Math.max(4, Number(item.importance) * 100)}%"></div>
                    </div>
                  </div>
                `)
                .join('')}
            </div>
          </div>
        ` : ''}
      </div>
    `)
    .join('');
}

function renderDownloadButtons(result) {
  if (!result.downloads) {
    downloadsSection.classList.add('d-none');
    downloadButtons.innerHTML = '';
    return;
  }

  const downloadItems = [
    { key: 'evaluation_report', label: 'Evaluation report', subtitle: 'CSV' },
    { key: 'model_comparison', label: 'Model comparison', subtitle: 'CSV' },
    { key: 'best_parameters', label: 'Best parameters', subtitle: 'JSON' },
    { key: 'best_model', label: 'Best model', subtitle: 'PKL' },
    { key: 'scaler', label: 'Scaler', subtitle: 'PKL' },
    { key: 'encoder', label: 'Encoder', subtitle: 'PKL' },
    { key: 'selected_features', label: 'Selected features', subtitle: 'PKL' },
    { key: 'dashboard_pdf', label: 'Dashboard report', subtitle: 'PDF' },
  ];

  downloadButtons.innerHTML = downloadItems
    .filter((item) => result.downloads[item.key])
    .map((item) => `
      <a class="btn btn-outline-light download-link" href="${result.downloads[item.key]}" download>
        <span class="d-block fw-semibold">${item.label}</span>
        <span class="small text-secondary">${item.subtitle}</span>
      </a>
    `)
    .join('');

  downloadsSection.classList.toggle('d-none', !downloadButtons.innerHTML);
}

function renderPredictionForm(result) {
  if (!predictionSection || !predictionFeatureGrid) {
    return;
  }

  const featureNames = Array.isArray(result.feature_names) ? result.feature_names : [];
  if (!featureNames.length) {
    predictionSection.classList.add('d-none');
    predictionFeatureGrid.innerHTML = '';
    return;
  }

  predictionFeatureGrid.innerHTML = featureNames
    .map((featureName) => `
      <div class="col-md-6 col-lg-4">
        <label class="form-label fw-semibold">${escapeHtml(featureName)}</label>
        <input class="form-control prediction-input" type="number" step="any" data-feature-name="${escapeHtml(featureName)}" placeholder="Enter value" />
      </div>
    `)
    .join('');

  predictionSection.classList.remove('d-none');
  predictionResultBox.innerHTML = '<div class="text-secondary">Enter values for the selected model features, then click Predict.</div>';
}

function renderPredictionResult(result, modelName) {
  const predictionValue = escapeHtml(result.prediction);
  const probabilityEntries = result.probabilities ? Object.entries(result.probabilities) : [];
  const probabilitiesMarkup = probabilityEntries.length
    ? `
      <div class="mt-3">
        <div class="small text-muted mb-2">Probabilities</div>
        <div class="d-grid gap-2">
          ${probabilityEntries.map(([label, probability]) => `
            <div class="d-flex justify-content-between"><span>${escapeHtml(label)}</span><strong>${(Number(probability) * 100).toFixed(1)}%</strong></div>
          `).join('')}
        </div>
      </div>
    `
    : '';

  predictionResultBox.innerHTML = `
    <div class="prediction-result-card">
      <div class="small text-muted mb-1">Model</div>
      <div class="fw-semibold mb-2">${escapeHtml(modelName || 'Saved model')}</div>
      <div class="small text-muted mb-1">Prediction</div>
      <div class="prediction-value">${predictionValue}</div>
      ${probabilitiesMarkup}
    </div>
  `;
}

async function runPrediction(event) {
  event.preventDefault();

  if (!window.currentJobId || !window.currentResult?.feature_names?.length) {
    setStatus('Train a model first so the saved prediction form can be loaded.', 'warning');
    showToast('Train a model first.', 'warning');
    return;
  }

  const featureInputs = predictionFeatureGrid?.querySelectorAll('[data-feature-name]') || [];
  const features = {};

  for (const input of featureInputs) {
    const featureName = input.dataset.featureName;
    const value = input.value.trim();
    if (value === '') {
      setStatus(`Please enter a value for ${featureName}.`, 'error');
      showToast(`Please enter a value for ${featureName}.`, 'danger');
      return;
    }

    features[featureName] = value;
  }

  setPredictionState(true);
  setStatus('Running prediction with the saved model...', 'info', 0);

  try {
    const response = await fetch(`${API_BASE_URL}/predict/${window.currentJobId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ features }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(payload, 'Prediction failed.'));
    }

    renderPredictionResult(payload, window.currentResult.best_model_name);
    setStatus('Prediction completed successfully.', 'success', 100);
    showToast('Prediction completed successfully.', 'success');
  } catch (error) {
    setStatus(error.message || 'Unable to run prediction.', 'error');
    showToast(error.message || 'Unable to run prediction.', 'danger');
  } finally {
    setPredictionState(false);
  }
}

function renderResult(result) {
  window.currentResult = result;
  window.currentJobId = result.job_id || window.currentJobId;

  const resultsSection = document.getElementById('results');
  const comparisonSection = document.getElementById('comparisonSection');
  const comparisonBreakdown = document.getElementById('comparisonBreakdown');
  const detailsSection = document.getElementById('details');

  resultsSection?.classList.remove('d-none');
  comparisonSection?.classList.remove('d-none');
  comparisonBreakdown?.classList.remove('d-none');
  detailsSection?.classList.remove('d-none');

  buildSummaryCards(result);
  renderChartTypeButtons();
  renderMetricButtons(result);
  selectedMetric = result.primary_metric || metricOptions[0];
  updateComparisonChart();
  updateScoreDistributionChart(result);
  updateFeatureImportanceChart(result);
  renderBestModel(result);
  renderTopModels(result);
  renderModelDetails(result);
  renderDownloadButtons(result);
  renderPredictionForm(result);
  // show preprocessing warnings if any
  try {
    const warnings = result.preprocessing && result.preprocessing.warnings;
    if (warnings && warnings.length) {
      setStatus(warnings.join('\n'), 'warning');
      showToast('Pipeline completed with preprocessing warnings.', 'warning');
    } else {
      setStatus('Pipeline completed successfully.', 'success', 100);
      showToast('Pipeline completed successfully. Downloads are ready.', 'success');
    }
  } catch (e) {
    setStatus('Pipeline completed successfully.', 'success', 100);
    showToast('Pipeline completed successfully. Downloads are ready.', 'success');
  }
}

async function runPipeline(event) {
  event.preventDefault();
  const file = document.getElementById('datasetFile').files[0];
  const targetColumn = document.getElementById('targetColumn').value.trim() || 'survived';

  if (!file) {
    setStatus('Please choose a dataset file before running the pipeline.', 'error');
    showToast('Please choose a dataset file before running the pipeline.', 'danger');
    return;
  }

  setRunningState(true);
  setStatus('Starting training pipeline...', 'info', 0);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('target_column', targetColumn);

  try {
    const response = await fetch(`${API_BASE_URL}/run-pipeline`, {
      method: 'POST',
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(payload, 'Pipeline execution failed.'));
    }

    window.currentJobId = payload.job_id;
    await pollPipelineStatus(payload.job_id);
  } catch (error) {
    clearPipelineStatusTimer();
    setRunningState(false);
    setStatus(error.message || 'Unable to run the pipeline.', 'error');
    showToast(error.message || 'Unable to run the pipeline.', 'danger');
  }
}

document.getElementById('pipelineForm').addEventListener('submit', runPipeline);
document.getElementById('predictionForm')?.addEventListener('submit', runPrediction);
