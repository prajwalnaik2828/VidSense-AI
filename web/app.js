/**
 * VidSense AI - Frontend Application Controller
 * Handles REST API interaction, WebSocket telemetry, live camera push, and responsive UI state.
 */

(function () {
  'use strict';

  // ---------------- Application State ----------------
  const state = {
    apiKey: localStorage.getItem('vidsense_api_key') || '',
    activeTab: 'search',
    sources: [],
    pollTimer: null,
    healthTimer: null,
    liveSessionId: null,
    liveMode: '0',
    liveWs: null,
    liveStreamInterval: null,
    localMediaStream: null,
    latestLiveDetections: [],
  };

  // ---------------- DOM Element References ----------------
  const dom = {
    // Nav & Status
    navTabs: document.querySelectorAll('.nav-tab'),
    sections: document.querySelectorAll('.tab-section'),
    serverStatusPill: document.getElementById('server-status-pill'),
    serverStatusText: document.getElementById('server-status-text'),
    badgeSourceCount: document.getElementById('badge-source-count'),
    btnApiKeyModal: document.getElementById('btn-api-key-modal'),

    // Search Tab
    formSearch: document.getElementById('form-search'),
    inputQuery: document.getElementById('input-query'),
    btnSearchClear: document.getElementById('btn-search-clear'),
    selectSearchSource: document.getElementById('select-search-source'),
    btnSearchSubmit: document.getElementById('btn-search-submit'),
    searchLoading: document.getElementById('search-loading'),
    searchResultsArea: document.getElementById('search-results-area'),
    cardAnswerBox: document.getElementById('card-answer-box'),
    badgeFoundStatus: document.getElementById('badge-found-status'),
    labelMatchCount: document.getElementById('label-match-count'),
    textAnswerContent: document.getElementById('text-answer-content'),
    gridSearchMatches: document.getElementById('grid-search-matches'),
    promptChips: document.querySelectorAll('.chip'),

    // Library Tab
    btnRefreshSources: document.getElementById('btn-refresh-sources'),
    dropzoneVideo: document.getElementById('dropzone-video'),
    inputVideoFile: document.getElementById('input-video-file'),
    uploadProgressCard: document.getElementById('upload-progress-card'),
    uploadFileName: document.getElementById('upload-file-name'),
    uploadPercentage: document.getElementById('upload-percentage'),
    uploadProgressBar: document.getElementById('upload-progress-bar'),
    uploadStatusDetail: document.getElementById('upload-status-detail'),
    gridSources: document.getElementById('grid-sources'),

    // Live Tab
    selectCameraMode: document.getElementById('select-camera-mode'),
    groupCustomPath: document.getElementById('group-custom-path'),
    inputCustomCameraPath: document.getElementById('input-custom-camera-path'),
    inputLiveName: document.getElementById('input-live-name'),
    btnStartLive: document.getElementById('btn-start-live'),
    btnStopLive: document.getElementById('btn-stop-live'),
    liveIndicatorDot: document.getElementById('live-indicator-dot'),
    liveSessionBadge: document.getElementById('live-session-badge'),
    liveSourceTitle: document.getElementById('live-source-title'),
    imgLiveMjpeg: document.getElementById('img-live-mjpeg'),
    videoLocalCamera: document.getElementById('video-local-camera'),
    canvasPushRender: document.getElementById('canvas-push-render'),
    livePlaceholderScreen: document.getElementById('live-placeholder-screen'),
    hudValFps: document.getElementById('hud-val-fps'),
    hudValTracks: document.getElementById('hud-val-tracks'),
    hudValElapsed: document.getElementById('hud-val-elapsed'),
    hudValCaptions: document.getElementById('hud-val-captions'),
    badgeEventSeq: document.getElementById('badge-event-seq'),
    liveEventsFeed: document.getElementById('live-events-feed'),

    // Modals
    modalClipPlayer: document.getElementById('modal-clip-player'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    videoClipElement: document.getElementById('video-clip-element'),
    modalClipTitle: document.getElementById('modal-clip-title'),
    modalClipMeta: document.getElementById('modal-clip-meta'),

    modalApiKey: document.getElementById('modal-api-key'),
    btnCloseKeyModal: document.getElementById('btn-close-key-modal'),
    inputApiKey: document.getElementById('input-api-key'),
    btnSaveApiKey: document.getElementById('btn-save-api-key'),

    toastContainer: document.getElementById('toast-container'),
  };

  // ---------------- HTTP Helpers ----------------
  function getHeaders(extraHeaders = {}) {
    const headers = { ...extraHeaders };
    if (state.apiKey) {
      headers['X-API-Key'] = state.apiKey;
    }
    return headers;
  }

  function appendApiKey(url) {
    if (!state.apiKey) return url;
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}api_key=${encodeURIComponent(state.apiKey)}`;
  }

  async function apiFetch(endpoint, options = {}) {
    options.headers = getHeaders(options.headers || {});
    const res = await fetch(endpoint, options);
    if (!res.ok) {
      let errMsg = `Request failed: ${res.status}`;
      try {
        const data = await res.json();
        if (data && data.error) errMsg = data.error;
      } catch (e) {}
      throw new Error(errMsg);
    }
    return res.json();
  }

  // ---------------- Notification Toasts ----------------
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // ---------------- Formatters ----------------
  function formatSeconds(secs) {
    if (!secs || isNaN(secs)) return '00:00';
    const total = Math.max(0, Math.floor(secs));
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  function formatDate(ts) {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // ---------------- Health & Server Status ----------------
  async function checkServerHealth() {
    try {
      const data = await fetch('/health').then(r => r.json());
      if (data && data.status === 'ok') {
        dom.serverStatusPill.className = 'status-pill status-online';
        dom.serverStatusText.textContent = 'Server Online';
      } else {
        throw new Error();
      }
    } catch (e) {
      dom.serverStatusPill.className = 'status-pill status-offline';
      dom.serverStatusText.textContent = 'Server Disconnected';
    }
  }

  // ---------------- Tab Navigation ----------------
  function switchTab(tabId) {
    state.activeTab = tabId;
    dom.navTabs.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    dom.sections.forEach(sec => {
      sec.classList.toggle('active', sec.id === `section-${tabId}`);
    });

    if (tabId === 'library') {
      loadSources();
    }
  }

  // ---------------- Video Sources Management ----------------
  async function loadSources() {
    try {
      const sources = await apiFetch('/sources');
      state.sources = sources;
      renderSources(sources);
      updateSearchSourceDropdown(sources);
      dom.badgeSourceCount.textContent = sources.length;

      // Check if any source is processing -> initiate polling
      const hasProcessing = sources.some(s => s.status === 'processing');
      if (hasProcessing && !state.pollTimer) {
        state.pollTimer = setInterval(loadSources, 2000);
      } else if (!hasProcessing && state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    } catch (err) {
      console.error('Failed to load sources:', err);
    }
  }

  function updateSearchSourceDropdown(sources) {
    const curVal = dom.selectSearchSource.value;
    dom.selectSearchSource.innerHTML = '<option value="">All Indexed Videos</option>';
    sources.forEach(s => {
      if (s.status === 'ready' || s.status === 'live') {
        const opt = document.createElement('option');
        opt.value = s.source_id;
        const statusLabel = s.status === 'live' ? 'LIVE NOW' : formatSeconds(s.duration_s);
        opt.textContent = `${s.name} (${s.kind}, ${statusLabel})`;
        dom.selectSearchSource.appendChild(opt);
      }
    });
    dom.selectSearchSource.value = curVal;
  }

  function renderSources(sources) {
    if (!sources || sources.length === 0) {
      dom.gridSources.innerHTML = `
        <div class="glass-card" style="grid-column: 1 / -1; padding: 2.5rem; text-align: center; color: var(--text-muted);">
          No videos indexed yet. Drop a video file above to begin tracking and indexing!
        </div>`;
      return;
    }

    dom.gridSources.innerHTML = '';
    sources.forEach(src => {
      const card = document.createElement('div');
      card.className = 'glass-card source-card';

      const classTags = Object.entries(src.class_counts || {})
        .map(([cls, cnt]) => `<span class="class-count-tag">${cls}: ${cnt}</span>`)
        .join('');

      card.innerHTML = `
        <div class="source-header-row">
          <span class="source-name" title="${src.name}">${src.name}</span>
          <span class="status-badge ${src.status}">${src.status}</span>
        </div>

        <div class="source-meta-grid">
          <div class="source-meta-item"><strong>Kind:</strong> ${src.kind}</div>
          <div class="source-meta-item"><strong>Tracks:</strong> ${src.n_tracks || 0}</div>
          <div class="source-meta-item"><strong>Duration:</strong> ${formatSeconds(src.duration_s)}</div>
          <div class="source-meta-item"><strong>Captions:</strong> ${src.n_captions || 0}</div>
        </div>

        ${src.status === 'processing' ? `
          <div class="progress-bar-track">
            <div class="progress-bar-fill" style="width: ${Math.round((src.progress || 0) * 100)}%;"></div>
          </div>
          <span style="font-size: 0.75rem; color: var(--text-dim);">Processing frames: ${Math.round((src.progress || 0) * 100)}%</span>
        ` : ''}

        ${classTags ? `<div class="class-tags-row">${classTags}</div>` : ''}

        <div class="source-card-actions">
          <button class="btn-source-action btn-source-search" data-id="${src.source_id}" type="button">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            Query
          </button>
          <button class="btn-source-action btn-source-delete" data-id="${src.source_id}" type="button">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
            Delete
          </button>
        </div>
      `;

      // Card action listeners
      card.querySelector('.btn-source-search').addEventListener('click', () => {
        dom.selectSearchSource.value = src.source_id;
        switchTab('search');
        dom.inputQuery.focus();
      });

      card.querySelector('.btn-source-delete').addEventListener('click', async () => {
        if (confirm(`Delete source "${src.name}" and remove all indexed tracks, clips, and vectors?`)) {
          try {
            await apiFetch(`/sources/${src.source_id}`, { method: 'DELETE' });
            showToast(`Deleted source "${src.name}"`, 'success');
            loadSources();
          } catch (err) {
            showToast(err.message, 'error');
          }
        }
      });

      dom.gridSources.appendChild(card);
    });
  }

  // ---------------- Video File Upload ----------------
  function setupUpload() {
    dom.dropzoneVideo.addEventListener('click', () => dom.inputVideoFile.click());

    dom.dropzoneVideo.addEventListener('dragover', (e) => {
      e.preventDefault();
      dom.dropzoneVideo.classList.add('dragover');
    });

    dom.dropzoneVideo.addEventListener('dragleave', () => {
      dom.dropzoneVideo.classList.remove('dragover');
    });

    dom.dropzoneVideo.addEventListener('drop', (e) => {
      e.preventDefault();
      dom.dropzoneVideo.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleUploadFile(e.dataTransfer.files[0]);
      }
    });

    dom.inputVideoFile.addEventListener('change', () => {
      if (dom.inputVideoFile.files && dom.inputVideoFile.files.length > 0) {
        handleUploadFile(dom.inputVideoFile.files[0]);
      }
    });
  }

  function handleUploadFile(file) {
    if (!file) return;

    dom.uploadProgressCard.style.display = 'block';
    dom.uploadFileName.textContent = file.name;
    dom.uploadPercentage.textContent = '0%';
    dom.uploadProgressBar.style.width = '0%';
    dom.uploadStatusDetail.textContent = 'Transferring video file to server...';

    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/sources/upload', true);

    if (state.apiKey) {
      xhr.setRequestHeader('X-API-Key', state.apiKey);
    }

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        dom.uploadPercentage.textContent = `${pct}%`;
        dom.uploadProgressBar.style.width = `${pct}%`;
      }
    };

    xhr.onload = () => {
      if (xhr.status === 202) {
        dom.uploadStatusDetail.textContent = 'Uploaded successfully! Background YOLO & scene indexing started.';
        showToast(`Video "${file.name}" uploaded. Processing started!`, 'success');
        dom.inputVideoFile.value = '';
        setTimeout(() => {
          dom.uploadProgressCard.style.display = 'none';
          loadSources();
        }, 1500);
      } else {
        let err = 'Upload failed';
        try {
          const res = JSON.parse(xhr.responseText);
          if (res && res.error) err = res.error;
        } catch (e) {}
        dom.uploadStatusDetail.textContent = `Error: ${err}`;
        showToast(err, 'error');
      }
    };

    xhr.onerror = () => {
      dom.uploadStatusDetail.textContent = 'Network error during upload.';
      showToast('Network error during upload', 'error');
    };

    xhr.send(formData);
  }

  // ---------------- Search & Query Flow ----------------
  async function executeSearch() {
    const query = dom.inputQuery.value.trim();
    if (!query) return;

    const sourceId = dom.selectSearchSource.value || null;

    dom.searchLoading.style.display = 'block';
    dom.searchResultsArea.style.display = 'none';
    dom.btnSearchSubmit.disabled = true;

    try {
      const data = await apiFetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, source_id: sourceId }),
      });

      renderSearchResults(data);
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      dom.searchLoading.style.display = 'none';
      dom.btnSearchSubmit.disabled = false;
    }
  }

  function renderSearchResults(data) {
    dom.searchResultsArea.style.display = 'block';

    const isFound = data.found;
    dom.badgeFoundStatus.className = isFound ? 'badge badge-success' : 'badge badge-dim';
    dom.badgeFoundStatus.textContent = isFound ? 'Answer Found' : 'Not Found';

    const results = data.results || [];
    dom.labelMatchCount.textContent = `${results.length} result(s)`;

    // Format timestamps in text as styled tags
    let formattedAnswer = (data.answer || '').replace(/\(t=(\d{2}:\d{2})\)/g, '<span class="ts-tag">t=$1</span>');
    dom.textAnswerContent.innerHTML = `<p>${formattedAnswer}</p>`;

    // Render matches
    dom.gridSearchMatches.innerHTML = '';
    if (results.length === 0) {
      dom.gridSearchMatches.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 2rem; text-align: center; color: var(--text-dim);">
          No matching tracks or clips returned.
        </div>`;
      return;
    }

    results.forEach(item => {
      const card = document.createElement('div');
      card.className = 'match-card';

      const thumbUrl = item.thumbnail_url ? appendApiKey(item.thumbnail_url) : null;
      const clipUrl = item.clip_url ? appendApiKey(item.clip_url) : null;
      const timeStr = `${formatSeconds(item.ts_start)} - ${formatSeconds(item.ts_end)}`;
      const durationS = Math.max(1, Math.round(item.ts_end - item.ts_start));

      card.innerHTML = `
        ${clipUrl ? `
          <div class="match-video-wrapper">
            <video 
              class="match-video-clip" 
              src="${clipUrl}" 
              controls 
              playsinline 
              preload="metadata"
              poster="${thumbUrl || ''}"
              title="${item.label} (${timeStr})"
            ></video>
            <span class="match-time-tag">${timeStr} (${durationS}s span)</span>
          </div>
        ` : `
          <div class="match-thumb-wrapper">
            ${thumbUrl ? `
              <img class="match-thumb-img" src="${thumbUrl}" alt="${item.label}" loading="lazy" />
            ` : `
              <div class="match-thumb-fallback">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="8.5" cy="8.5" r="1.5"></circle>
                  <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
              </div>
            `}
            <span class="match-time-tag">${timeStr}</span>
          </div>
        `}

        <div class="match-content">
          <div class="match-label">${item.label}</div>
          <div class="match-source-meta">${item.source_name || item.source_id} &bull; ${item.kind}</div>
          ${clipUrl ? `
            <div class="match-actions">
              <button class="btn-clip-play" data-clip="${item.clip_url}" data-title="${item.label} (${timeStr})" type="button">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                Watch In Theater
              </button>
            </div>
          ` : ''}
        </div>
      `;

      // Hover preview behavior for the embedded clip
      if (clipUrl) {
        const videoEl = card.querySelector('.match-video-clip');
        if (videoEl) {
          card.addEventListener('mouseenter', () => {
            if (videoEl.paused) {
              videoEl.muted = true;
              videoEl.play().catch(() => {});
            }
          });
          card.addEventListener('mouseleave', () => {
            if (!videoEl.paused && videoEl.muted) {
              videoEl.pause();
            }
          });
        }

        const btnTheater = card.querySelector('.btn-clip-play');
        if (btnTheater) {
          btnTheater.addEventListener('click', (e) => {
            const btn = e.currentTarget;
            openClipModal(btn.dataset.clip, btn.dataset.title);
          });
        }
      }

      dom.gridSearchMatches.appendChild(card);
    });

    // Scroll to results
    dom.searchResultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ---------------- Clip Modal ----------------
  function openClipModal(clipPath, title) {
    const fullUrl = appendApiKey(clipPath);
    dom.videoClipElement.src = fullUrl;
    dom.modalClipTitle.textContent = title || 'Video Clip';
    dom.modalClipMeta.textContent = `Source URL: ${clipPath}`;
    dom.modalClipPlayer.style.display = 'flex';
    dom.videoClipElement.play().catch(() => {});
  }

  function closeClipModal() {
    dom.modalClipPlayer.style.display = 'none';
    dom.videoClipElement.pause();
    dom.videoClipElement.src = '';
  }

  // ---------------- Live Studio ----------------
  function setupLiveControls() {
    dom.selectCameraMode.addEventListener('change', () => {
      const mode = dom.selectCameraMode.value;
      if (mode === 'file' || mode === 'stream') {
        dom.groupCustomPath.style.display = 'block';
        dom.inputCustomCameraPath.placeholder = mode === 'file' ? 'e.g. tests/videos/sample.mp4' : 'e.g. rtsp://192.168.1.100:554/stream';
      } else {
        dom.groupCustomPath.style.display = 'none';
      }
    });

    dom.btnStartLive.addEventListener('click', startLiveSession);
    dom.btnStopLive.addEventListener('click', stopLiveSession);
  }

  async function startLiveSession() {
    const mode = dom.selectCameraMode.value;
    let cameraParam = mode;
    if (mode === 'file' || mode === 'stream') {
      cameraParam = dom.inputCustomCameraPath.value.trim();
      if (!cameraParam) {
        showToast('Please specify a file path or stream URL', 'error');
        return;
      }
    }

    const sessionName = dom.inputLiveName.value.trim() || 'Live Camera';

    dom.btnStartLive.disabled = true;
    dom.btnStartLive.textContent = 'Connecting...';

    try {
      // 1. Start live session on backend
      const res = await apiFetch('/live/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera: cameraParam, name: sessionName }),
      });

      const sourceId = res.source_id;
      state.liveSessionId = sourceId;
      state.liveMode = mode;

      // 2. Update UI to Live State
      dom.btnStartLive.style.display = 'none';
      dom.btnStopLive.style.display = 'inline-flex';
      dom.liveIndicatorDot.className = 'live-pulse-dot active';
      dom.liveSessionBadge.textContent = 'LIVE ON AIR';
      dom.liveSessionBadge.style.color = 'var(--accent-emerald)';
      dom.liveSourceTitle.textContent = `${sessionName} (${sourceId})`;
      dom.livePlaceholderScreen.style.display = 'none';
      dom.liveEventsFeed.innerHTML = '';

      showToast(`Live session started (${sourceId})`, 'success');

      // 3. Attach Stream & WebSocket
      if (mode === 'push') {
        await startBrowserPushMode(sourceId);
      } else {
        startServerStreamMode(sourceId);
      }

      connectLiveWebSocket(sourceId);
    } catch (err) {
      showToast(err.message, 'error');
      dom.btnStartLive.style.display = 'inline-flex';
      dom.btnStopLive.style.display = 'none';
    } finally {
      dom.btnStartLive.disabled = false;
      dom.btnStartLive.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Start Live Session`;
    }
  }

  function startServerStreamMode(sourceId) {
    dom.canvasPushRender.style.display = 'none';
    dom.videoLocalCamera.style.display = 'none';
    dom.imgLiveMjpeg.style.display = 'block';
    dom.imgLiveMjpeg.src = appendApiKey(`/live/${sourceId}/stream`);
  }

  async function startBrowserPushMode(sourceId) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { max: 15 } },
        audio: false,
      });
      state.localMediaStream = stream;

      dom.imgLiveMjpeg.style.display = 'none';
      dom.videoLocalCamera.style.display = 'none'; // Keep hidden, draw to canvas
      dom.canvasPushRender.style.display = 'block';

      dom.videoLocalCamera.srcObject = stream;
      await dom.videoLocalCamera.play();

      const canvas = dom.canvasPushRender;
      const ctx = canvas.getContext('2d');
      canvas.width = 640;
      canvas.height = 480;

      // Hidden capture canvas for resizing to 480px width
      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = 480;
      captureCanvas.height = 360;
      const captureCtx = captureCanvas.getContext('2d');

      // Periodic frame pusher: 5 FPS (200 ms)
      state.liveStreamInterval = setInterval(() => {
        if (!state.liveWs || state.liveWs.readyState !== WebSocket.OPEN) return;

        // Render video frame to local canvas with latest detections
        ctx.drawImage(dom.videoLocalCamera, 0, 0, canvas.width, canvas.height);
        drawBoundingBoxes(ctx, state.latestLiveDetections, canvas.width, canvas.height);

        // Render resized frame to capture canvas for JPEG blob push
        captureCtx.drawImage(dom.videoLocalCamera, 0, 0, captureCanvas.width, captureCanvas.height);
        captureCanvas.toBlob((blob) => {
          if (blob && state.liveWs && state.liveWs.readyState === WebSocket.OPEN) {
            state.liveWs.send(blob);
          }
        }, 'image/jpeg', 0.75);
      }, 200);

    } catch (err) {
      showToast(`Camera permission error: ${err.message}`, 'error');
      stopLiveSession();
    }
  }

  function drawBoundingBoxes(ctx, detections, w, h) {
    if (!detections || detections.length === 0) return;

    detections.forEach(det => {
      const box = det.bbox || [0, 0, 0, 0];
      const scaleX = w / 480.0;
      const scaleY = h / 360.0;
      const x1 = box[0] * scaleX;
      const y1 = box[1] * scaleY;
      const x2 = box[2] * scaleX;
      const y2 = box[3] * scaleY;

      // Box
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label
      const label = `${det.cls} #${det.track_id} (${Math.round(det.conf * 100)}%)`;
      ctx.font = '12px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = '#10b981';
      ctx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);

      ctx.fillStyle = '#000000';
      ctx.fillText(label, x1 + 4, Math.max(14, y1 - 4));
    });
  }

  function connectLiveWebSocket(sourceId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = appendApiKey(`${protocol}//${window.location.host}/live/${sourceId}/ws`);

    const ws = new WebSocket(wsUrl);
    state.liveWs = ws;
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          handleLiveEvent(msg);
        } catch (e) {}
      }
    };

    ws.onclose = () => {
      console.log('Live WebSocket closed');
    };

    ws.onerror = (e) => {
      console.error('Live WebSocket error:', e);
    };
  }

  function handleLiveEvent(msg) {
    if (msg.seq) {
      dom.badgeEventSeq.textContent = `Seq: ${msg.seq}`;
    }

    if (msg.stats) {
      dom.hudValFps.textContent = (msg.stats.fps || 0).toFixed(1);
      dom.hudValTracks.textContent = msg.stats.active_tracks || 0;
      dom.hudValElapsed.textContent = formatSeconds(msg.stats.elapsed_s);
      dom.hudValCaptions.textContent = msg.stats.captions || 0;
    }

    const detections = msg.detections || [];
    state.latestLiveDetections = detections;

    if (detections.length > 0) {
      renderEventFeedItem(msg);
    }
  }

  function renderEventFeedItem(msg) {
    const item = document.createElement('div');
    item.className = 'event-entry';

    const chips = (msg.detections || [])
      .map(d => `<span class="det-chip">${d.cls} #${d.track_id} (${Math.round(d.conf * 100)}%)</span>`)
      .join('');

    item.innerHTML = `
      <div class="event-entry-top">
        <span>t = ${formatSeconds(msg.ts)}</span>
        <span>Seq #${msg.seq}</span>
      </div>
      <div class="event-chips">${chips}</div>
    `;

    dom.liveEventsFeed.prepend(item);
    if (dom.liveEventsFeed.children.length > 30) {
      dom.liveEventsFeed.lastElementChild.remove();
    }
  }

  async function stopLiveSession() {
    if (!state.liveSessionId) return;

    dom.btnStopLive.disabled = true;
    dom.btnStopLive.textContent = 'Stopping & Flushing...';

    // Stop browser camera if active
    if (state.liveStreamInterval) {
      clearInterval(state.liveStreamInterval);
      state.liveStreamInterval = null;
    }
    if (state.localMediaStream) {
      state.localMediaStream.getTracks().forEach(t => t.stop());
      state.localMediaStream = null;
    }
    if (state.liveWs) {
      state.liveWs.close();
      state.liveWs = null;
    }

    try {
      const data = await apiFetch(`/live/${state.liveSessionId}/stop`, { method: 'POST' });
      showToast(`Live session finalized! Recorded ${data.n_tracks} tracks (${formatSeconds(data.duration_s)}).`, 'success');
      loadSources();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      state.liveSessionId = null;
      dom.btnStartLive.style.display = 'inline-flex';
      dom.btnStopLive.style.display = 'none';
      dom.btnStopLive.disabled = false;
      dom.btnStopLive.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="6" y="6" width="12" height="12"></rect>
        </svg>
        Stop & Save Session`;

      dom.liveIndicatorDot.className = 'live-pulse-dot';
      dom.liveSessionBadge.textContent = 'OFFLINE';
      dom.liveSessionBadge.style.color = 'var(--text-muted)';
      dom.liveSourceTitle.textContent = 'Session finished & indexed';

      dom.imgLiveMjpeg.src = '';
      dom.imgLiveMjpeg.style.display = 'none';
      dom.canvasPushRender.style.display = 'none';
      dom.livePlaceholderScreen.style.display = 'flex';
    }
  }

  // ---------------- API Key Modal ----------------
  function setupApiKeyModal() {
    dom.btnApiKeyModal.addEventListener('click', () => {
      dom.inputApiKey.value = state.apiKey;
      dom.modalApiKey.style.display = 'flex';
    });

    dom.btnCloseKeyModal.addEventListener('click', () => {
      dom.modalApiKey.style.display = 'none';
    });

    dom.btnSaveApiKey.addEventListener('click', () => {
      state.apiKey = dom.inputApiKey.value.trim();
      if (state.apiKey) {
        localStorage.setItem('vidsense_api_key', state.apiKey);
        showToast('API key updated and saved', 'success');
      } else {
        localStorage.removeItem('vidsense_api_key');
        showToast('API key cleared', 'info');
      }
      dom.modalApiKey.style.display = 'none';
      loadSources();
      checkServerHealth();
    });
  }

  // ---------------- Event Listeners Setup ----------------
  function initEventListeners() {
    // Navigation
    dom.navTabs.forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Search
    dom.formSearch.addEventListener('submit', (e) => {
      e.preventDefault();
      executeSearch();
    });

    dom.inputQuery.addEventListener('input', () => {
      dom.btnSearchClear.style.display = dom.inputQuery.value ? 'block' : 'none';
    });

    dom.btnSearchClear.addEventListener('click', () => {
      dom.inputQuery.value = '';
      dom.btnSearchClear.style.display = 'none';
      dom.inputQuery.focus();
    });

    dom.promptChips.forEach(chip => {
      chip.addEventListener('click', () => {
        dom.inputQuery.value = chip.dataset.prompt;
        dom.btnSearchClear.style.display = 'block';
        executeSearch();
      });
    });

    // Modals
    dom.btnCloseModal.addEventListener('click', closeClipModal);
    dom.modalClipPlayer.addEventListener('click', (e) => {
      if (e.target === dom.modalClipPlayer) closeClipModal();
    });
    dom.modalApiKey.addEventListener('click', (e) => {
      if (e.target === dom.modalApiKey) dom.modalApiKey.style.display = 'none';
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeClipModal();
        dom.modalApiKey.style.display = 'none';
      }
    });

    dom.btnRefreshSources.addEventListener('click', () => {
      loadSources();
      showToast('Sources refreshed', 'info');
    });

    setupUpload();
    setupLiveControls();
    setupApiKeyModal();
  }

  // ---------------- App Initialization ----------------
  function init() {
    initEventListeners();
    checkServerHealth();
    loadSources();
    state.healthTimer = setInterval(checkServerHealth, 5000);
  }

  // Launch when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
