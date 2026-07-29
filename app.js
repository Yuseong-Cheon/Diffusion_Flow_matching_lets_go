/**
 * Generative Model Explainer Engine
 * Flow Matching conditional paths and a score-based reverse-SDE toy model.
 */

function mulberry32(a) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

function normalPair(rng) {
  const radius = Math.sqrt(-2 * Math.log(Math.max(rng(), Number.EPSILON)));
  const angle = 2 * Math.PI * rng();
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

class ExplainerStudio {
  constructor() {
    this.canvas = document.getElementById('manifold-canvas');
    this.ctx = this.canvas.getContext('2d');
    
    this.targetCanvas = document.getElementById('target-img-canvas');
    this.targetCtx = this.targetCanvas.getContext('2d');

    // UI Elements
    this.promptSelect = document.getElementById('prompt-select');
    this.seedSlider = document.getElementById('seed-slider');
    this.seedBadge = document.getElementById('seed-badge');
    this.btnRandomSeed = document.getElementById('btn-random-seed');
    this.noiseGridContainer = document.getElementById('noise-matrix-grid');
    
    this.selectedCellName = document.getElementById('selected-cell-name');
    this.selectedCellVal = document.getElementById('selected-cell-val');
    this.selectedCellProjection = document.getElementById('selected-cell-projection');
    this.selectedCellTarget = document.getElementById('selected-cell-target');
    this.targetBreedBadge = document.getElementById('target-breed-badge');
    this.predictedX0Status = document.getElementById('predicted-x0-status');
    
    // Velocity Vector Metric DOM
    this.metricRgbZt = document.getElementById('metric-rgb-zt');
    this.metricVectorVal = document.getElementById('metric-vector-val');
    this.metricVectorStatus = document.getElementById('metric-vector-status');
    this.metricVectorLabel = document.getElementById('metric-vector-label');
    this.metricNearestName = document.getElementById('metric-nearest-name');
    this.batchMeanPreview = document.getElementById('batch-mean-preview');
    this.batchMeanCanvas = document.getElementById('batch-mean-canvas');
    this.batchMeanBadge = document.getElementById('batch-mean-badge');
    this.modeBoundaryRow = document.getElementById('mode-boundary-row');
    this.modeBoundaryStatus = document.getElementById('mode-boundary-status');
    this.metricConfidence = document.getElementById('metric-confidence');
    this.metricConfidenceLabel = document.getElementById('metric-confidence-label');
    this.metricConfBar = document.getElementById('metric-conf-bar');
    this.metricConfidenceRow = document.getElementById('metric-confidence-row');
    this.metricConfidenceBar = document.getElementById('metric-confidence-bar');
    this.modeResponsibilityList = document.getElementById('mode-responsibility-list');
    this.modeResponsibilityBars = document.getElementById('mode-responsibility-bars');
    this.diffusionStepBreakdown = document.getElementById('diffusion-step-breakdown');
    this.metricDriftStep = document.getElementById('metric-drift-step');
    this.metricNoiseStep = document.getElementById('metric-noise-step');
    this.metricTotalStep = document.getElementById('metric-total-step');

    this.btnModeFlow = document.getElementById('btn-mode-flow');
    this.btnModeDiff = document.getElementById('btn-mode-diff');

    this.btnPlayPause = document.getElementById('btn-play-pause');
    this.btnReset = document.getElementById('btn-reset');
    this.scrubber = document.getElementById('scrubber');
    this.diffusionStepSelect = document.getElementById('diffusion-step-select');
    this.btnResampleDiffusion = document.getElementById('btn-resample-diffusion');
    this.diffusionRunLabel = document.getElementById('diffusion-run-label');
    this.speedSelect = document.getElementById('speed-select');
    this.btnExportVideo = document.getElementById('btn-export-video');

    // State Parameters
    this.seed = parseInt(this.seedSlider.value) || 42;
    this.prompt = this.promptSelect.value || 'cat';
    this.algorithm = 'flow';
    this.currentTime = 0;
    this.isPlaying = true;
    this.speed = parseFloat(this.speedSelect.value) || 1.0;
    this.selectedCellIndex = 0;
    this.diffusionNoiseRun = 0;
    this.dataModeSigma = 0.18;
    this.modeBoundarySigma = 2.5;

    // Data Structures
    this.noiseGrid = [];
    this.winningBreed = null;
    
    // Video Recorder
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.isRecording = false;
    this.recBadge = document.getElementById('rec-badge');
    this.recTimer = document.getElementById('rec-timer');

    this.initCanvasSize();
    this.setupEventListeners();
    this.generateSeedData();
    
    // Start Animation Loop
    this.lastTimestamp = performance.now();
    requestAnimationFrame((ts) => this.loop(ts));
  }

  initCanvasSize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
  }

  setupEventListeners() {
    window.addEventListener('resize', () => {
      this.initCanvasSize();
      this.generateSeedData();
    });

    this.promptSelect.addEventListener('change', (e) => {
      this.prompt = e.target.value;
      this.diffusionNoiseRun = 0;
      this.diffusionRunLabel.textContent = 'SDE noise #1';
      this.generateSeedData();
    });

    this.seedSlider.addEventListener('input', (e) => {
      this.seed = parseInt(e.target.value);
      this.seedBadge.textContent = `시드 #${this.seed}`;
      this.diffusionNoiseRun = 0;
      this.diffusionRunLabel.textContent = 'SDE noise #1';
      this.generateSeedData();
    });

    this.btnRandomSeed.addEventListener('click', () => {
      this.seed = Math.floor(Math.random() * 999) + 1;
      this.seedSlider.value = this.seed;
      this.seedBadge.textContent = `시드 #${this.seed}`;
      this.diffusionNoiseRun = 0;
      this.diffusionRunLabel.textContent = 'SDE noise #1';
      this.generateSeedData();
    });

    this.btnModeFlow.addEventListener('click', () => {
      this.algorithm = 'flow';
      this.btnModeFlow.classList.add('active');
      this.btnModeDiff.classList.remove('active');
      this.predictedX0Status.textContent = "선택한 (z₀, x₁) 쌍의 조건부 target uτ를 따라 직선으로 이동합니다. 학습된 주변 속도장은 일반적으로 위치와 시간에 따라 달라집니다.";
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    });

    this.btnModeDiff.addEventListener('click', () => {
      this.algorithm = 'diff';
      this.btnModeDiff.classList.add('active');
      this.btnModeFlow.classList.remove('active');
      this.predictedX0Status.textContent = "Gaussian-mixture의 정확한 score로 reverse SDE를 적분합니다. 확률적 경로는 흔들리지만 score가 데이터 mode로 이끕니다.";
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    });

    this.scrubber.addEventListener('input', (e) => {
      this.currentTime = parseInt(e.target.value) / 1000;
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    });

    this.speedSelect.addEventListener('change', (e) => {
      this.speed = parseFloat(e.target.value);
    });

    this.diffusionStepSelect.addEventListener('change', () => {
      this.buildDiffusionPaths();
      this.selfCheck();
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    });

    this.btnResampleDiffusion.addEventListener('click', () => {
      this.diffusionNoiseRun++;
      this.diffusionRunLabel.textContent = `SDE noise #${this.diffusionNoiseRun + 1}`;
      this.currentTime = 0;
      this.scrubber.value = 0;
      this.buildDiffusionPaths();
      this.selfCheck();
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    });

    this.btnPlayPause.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      this.btnPlayPause.innerHTML = this.isPlaying ? 
        '<i data-lucide="pause"></i>' : '<i data-lucide="play"></i>';
      if (window.lucide) { lucide.createIcons(); }
    });

    this.btnReset.addEventListener('click', () => {
      this.currentTime = 0;
      this.scrubber.value = 0;
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    });

    this.btnExportVideo.addEventListener('click', () => {
      this.toggleRecording();
    });
  }

  generateSeedData() {
    const rng = mulberry32(this.seed * 9999 + (this.prompt === 'cat' ? 100 : 200));
    this.noiseGrid = [];

    const richPools = {
      cat: [
        { id: 0, name: '페르시안 고양이', shortName: '페르시안', color: '#e2e8f0', pattern: 'fluffy' },
        { id: 1, name: '치즈 냥이', shortName: '치즈 냥이', color: '#f59e0b', pattern: 'stripes' },
        { id: 2, name: '검은 고양이', shortName: '검은 냥이', color: '#334155', pattern: 'dark' },
        { id: 3, name: '러시안 블루', shortName: '러시안블루', color: '#6366f1', pattern: 'blue' },
        { id: 4, name: '삼색이', shortName: '삼색이', color: '#ec4899', pattern: 'calico' },
        { id: 5, name: '샴 고양이', shortName: '샴 냥이', color: '#fcd34d', pattern: 'mask' },
        { id: 6, name: '스핑크스 냥이', shortName: '스핑크스', color: '#f43f5e', pattern: 'pink' },
        { id: 7, name: '뱅갈 고양이', shortName: '뱅갈 냥이', color: '#eab308', pattern: 'spots' },
        { id: 8, name: '코리안 숏헤어', shortName: '코숏 냥이', color: '#38bdf8', pattern: 'short' },
        { id: 9, name: '스코티시 폴드', shortName: '스코티시', color: '#a855f7', pattern: 'folded' },
        { id: 10, name: '메인쿤 고양이', shortName: '메인쿤', color: '#d97706', pattern: 'fluffy' },
        { id: 11, name: '렉돌 고양이', shortName: '렉돌 냥이', color: '#38bdf8', pattern: 'mask' },
        { id: 12, name: '먼치킨 고양이', shortName: '먼치킨', color: '#f43f5e', pattern: 'short' },
        { id: 13, name: '브리티시 숏헤어', shortName: '브리티시', color: '#94a3b8', pattern: 'blue' },
        { id: 14, name: '아비시니안', shortName: '아비시니안', color: '#ca8a04', pattern: 'stripes' },
        { id: 15, name: '노르웨이 숲', shortName: '노르웨이숲', color: '#16a34a', pattern: 'fluffy' },
        { id: 16, name: '터키시 앙고라', shortName: '터키시앙고라', color: '#ffffff', pattern: 'fluffy' },
        { id: 17, name: '버만 고양이', shortName: '버만 냥이', color: '#fb7185', pattern: 'mask' },
        { id: 18, name: '샤르트뢰 고양이', shortName: '샤르트뢰', color: '#475569', pattern: 'blue' },
        { id: 19, name: '엑조틱 숏헤어', shortName: '엑조틱', color: '#f97316', pattern: 'folded' }
      ],
      dog: [
        { id: 0, name: '시바견', shortName: '시바견', color: '#f59e0b', pattern: 'shiba' },
        { id: 1, name: '골든 리트리버', shortName: '리트리버', color: '#eab308', pattern: 'golden' },
        { id: 2, name: '포메라니안', shortName: '포메라니안', color: '#ec4899', pattern: 'pome' },
        { id: 3, name: '시베리안 허스키', shortName: '허스키', color: '#38bdf8', pattern: 'husky' },
        { id: 4, name: '토이 푸들', shortName: '토이푸들', color: '#a855f7', pattern: 'poodle' },
        { id: 5, name: '비숑 프리제', shortName: '비숑프리제', color: '#ffffff', pattern: 'bichon' },
        { id: 6, name: '웰시 코기', shortName: '웰시코기', color: '#f97316', pattern: 'corgi' },
        { id: 7, name: '닥스훈트', shortName: '닥스훈트', color: '#78350f', pattern: 'long' },
        { id: 8, name: '달마시안', shortName: '달마시안', color: '#06b6d4', pattern: 'spots' },
        { id: 9, name: '말티즈', shortName: '말티즈', color: '#f8fafc', pattern: 'bichon' },
        { id: 10, name: '프렌치 불독', shortName: '프렌치불독', color: '#475569', pattern: 'shiba' },
        { id: 11, name: '슈나우저', shortName: '슈나우저', color: '#64748b', pattern: 'husky' },
        { id: 12, name: '사모예드', shortName: '사모예드', color: '#ffffff', pattern: 'pome' },
        { id: 13, name: '셰퍼드', shortName: '셰퍼드', color: '#9a3412', pattern: 'shiba' },
        { id: 14, name: '비글', shortName: '비글', color: '#ea580c', pattern: 'corgi' },
        { id: 15, name: '보더 콜리', shortName: '보더콜리', color: '#1e293b', pattern: 'husky' },
        { id: 16, name: '치와와', shortName: '치와와', color: '#fde047', pattern: 'pome' },
        { id: 17, name: '요크셔 테리어', shortName: '요크셔', color: '#ca8a04', pattern: 'poodle' },
        { id: 18, name: '도베르만', shortName: '도베르만', color: '#090d16', pattern: 'shiba' },
        { id: 19, name: '스피츠', shortName: '스피츠', color: '#f1f5f9', pattern: 'pome' }
      ]
    };

    const fullPool = richPools[this.prompt];
    const w = this.canvas.width;
    const h = this.canvas.height;

    const promptCenters = {
      cat: { x: w * 0.65, y: h * 0.45, radiusX: w * 0.162, radiusY: h * 0.290 },
      dog: { x: w * 0.62, y: h * 0.72, radiusX: w * 0.148, radiusY: h * 0.266 }
    };
    const classCenter = promptCenters[this.prompt];

    // EVERY SINGLE BREED IN THE 20-SPECIES POOL HAS ITS OWN UNIQUE DEDICATED (x, y) POSITION ON THE MANIFOLD!
    const allBreedFixedClusters = fullPool.map((breed) => {
      const angle = (breed.id / 20) * Math.PI * 2;
      const radiusFactor = 0.55 + ((breed.id % 3) * 0.15);
      return {
        ...breed,
        breedName: breed.name,
        targetColor: breed.color,
        x1: {
          x: classCenter.x + Math.cos(angle) * classCenter.radiusX * radiusFactor,
          y: classCenter.y + Math.sin(angle) * classCenter.radiusY * radiusFactor
        }
      };
    });

    const shuffledPool = [...allBreedFixedClusters];
    for (let i = shuffledPool.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffledPool[i], shuffledPool[j]] = [shuffledPool[j], shuffledPool[i]];
    }
    const sampledPool = shuffledPool.slice(0, 9);
    const priorBounds = this.getPriorBounds();
    const center = { x: priorBounds.x + priorBounds.width / 2, y: priorBounds.y + priorBounds.height / 2 };
    const noiseScale = Math.min(priorBounds.width, priorBounds.height) * 0.18;
    const maxLatent = priorBounds.width * 0.42 / noiseScale;

    for (let i = 0; i < 9; i++) {
      let noiseR, noiseG, noiseB, noiseX, noiseY;
      do {
        [noiseR, noiseG] = normalPair(rng);
        [noiseB] = normalPair(rng);
        noiseX = (noiseR - noiseG) / Math.sqrt(2);
        noiseY = (noiseR + noiseG - 2 * noiseB) / Math.sqrt(6);
      } while (Math.abs(noiseX) > maxLatent || Math.abs(noiseY) > maxLatent);
      const r = Math.max(0, Math.min(255, Math.round(128 + noiseR * 45)));
      const g = Math.max(0, Math.min(255, Math.round(128 + noiseG * 45)));
      const b = Math.max(0, Math.min(255, Math.round(128 + noiseB * 45)));

      const targetBreed = sampledPool[i];

      this.noiseGrid.push({
        name: String.fromCharCode(65 + i),
        fullName: targetBreed.name,
        targetShortName: targetBreed.shortName,
        r, g, b,
        rgbNoise: { r: noiseR, g: noiseG, b: noiseB },
        noiseValue: { x: noiseX, y: noiseY },
        z0: null,
        x1: targetBreed.x1,
        breedName: targetBreed.name,
        targetColor: targetBreed.color,
        pattern: targetBreed.pattern
      });
    }

    this.noiseScaleX = noiseScale;
    this.noiseScaleY = noiseScale;
    this.noiseGrid.forEach(cell => {
      cell.z0 = {
        x: center.x + cell.noiseValue.x * this.noiseScaleX,
        y: center.y + cell.noiseValue.y * this.noiseScaleY
      };
    });

    this.fixedBreedMap = sampledPool;
    this.allBreedFixedClusters = allBreedFixedClusters;
    this.buildDiffusionPaths();
    this.selfCheck();

    this.renderNoiseGridDOM();
    this.updateSelectedCellDetail();
    this.renderTargetImage();
    this.renderEvolutionCanvases();
  }

  renderNoiseGridDOM() {
    this.noiseGridContainer.innerHTML = '';
    this.noiseGrid.forEach((cell, idx) => {
      const cellEl = document.createElement('div');
      cellEl.className = `matrix-cell ${idx === this.selectedCellIndex ? 'selected' : ''}`;
      cellEl.style.backgroundColor = `rgb(${cell.r}, ${cell.g}, ${cell.b})`;
      
      cellEl.innerHTML = `
        <span class="cell-name">${cell.name} (${cell.targetShortName})</span>
        <span class="cell-val">[${cell.rgbNoise.r >= 0 ? '+' : ''}${cell.rgbNoise.r.toFixed(1)},${cell.rgbNoise.g >= 0 ? '+' : ''}${cell.rgbNoise.g.toFixed(1)},${cell.rgbNoise.b >= 0 ? '+' : ''}${cell.rgbNoise.b.toFixed(1)}]</span>
      `;

      cellEl.addEventListener('click', () => {
        this.selectedCellIndex = idx;
        this.renderNoiseGridDOM();
        this.updateSelectedCellDetail();
        this.renderTargetImage();
      });

      this.noiseGridContainer.appendChild(cellEl);
    });
  }

  getDiffusionStats(pos, generationTime) {
    const center = { x: this.canvas.width * 0.12, y: this.canvas.height * 0.50 };
    const scaleX = this.noiseScaleX;
    const scaleY = this.noiseScaleY;
    const y = { x: (pos.x - center.x) / scaleX, y: (pos.y - center.y) / scaleY };
    const s = 1 - generationTime;
    const betaMin = 0.2;
    const betaMax = 12;
    const beta = betaMin + (betaMax - betaMin) * s;
    const integratedBeta = betaMin * s + 0.5 * (betaMax - betaMin) * s * s;
    const alpha = Math.exp(-0.5 * integratedBeta);
    const dataVariance = this.dataModeSigma ** 2;
    const variance = alpha * alpha * dataVariance + (1 - alpha * alpha);

    const components = this.allBreedFixedClusters.map((breed) => {
      const meanX = alpha * (breed.x1.x - center.x) / scaleX;
      const meanY = alpha * (breed.x1.y - center.y) / scaleY;
      const distanceSquared = (y.x - meanX) ** 2 + (y.y - meanY) ** 2;
      return { breed, meanX, meanY, logWeight: -distanceSquared / (2 * variance) };
    });
    const maxLogWeight = Math.max(...components.map(component => component.logWeight));
    let totalWeight = 0;
    let weightedMeanX = 0;
    let weightedMeanY = 0;
    let bestComponent = components[0];
    let bestWeight = -1;

    components.forEach(component => {
      const weight = Math.exp(component.logWeight - maxLogWeight);
      component.weight = weight;
      totalWeight += weight;
      weightedMeanX += weight * component.meanX;
      weightedMeanY += weight * component.meanY;
      if (weight > bestWeight) {
        bestWeight = weight;
        bestComponent = component;
      }
    });

    return {
      y,
      beta,
      score: {
        x: (weightedMeanX / totalWeight - y.x) / variance,
        y: (weightedMeanY / totalWeight - y.y) / variance
      },
      breed: bestComponent.breed,
      responsibility: bestWeight / totalWeight,
      modeResponsibilities: components
        .map(component => ({ breed: component.breed, responsibility: component.weight / totalWeight }))
        .sort((a, b) => b.responsibility - a.responsibility),
      scaleX,
      scaleY
    };
  }

  getModeDistances(pos) {
    return this.allBreedFixedClusters
      .map(breed => ({
        breed,
        distance: Math.hypot(
          (pos.x - breed.x1.x) / this.noiseScaleX,
          (pos.y - breed.x1.y) / this.noiseScaleY
        )
      }))
      .sort((a, b) => a.distance - b.distance);
  }

  getModeMembership(pos) {
    const boundaryRadius = this.dataModeSigma * this.modeBoundarySigma;
    return this.getModeDistances(pos).filter(item => item.distance <= boundaryRadius);
  }

  getFinalModeStatus(pos) {
    const matches = this.getModeMembership(pos);
    if (!matches.length) {
      const boundaryRadius = this.dataModeSigma * this.modeBoundarySigma;
      const excess = this.getModeDistances(pos)[0].distance - boundaryRadius;
      const outsideWeight = Math.min(0.85, 1 - Math.exp(-0.5 * (excess / boundaryRadius) ** 2));
      return {
        type: 'outside',
        label: `기타 ${Math.round(outsideWeight * 100)}% · 2.5σ 경계 밖`,
        outsideWeight,
        matches
      };
    }
    if (matches.length === 1) {
      const breed = matches[0].breed;
      return { type: 'single', label: breed.shortName || breed.name, breed, matches };
    }
    const names = matches.slice(0, 2).map(item => item.breed.shortName || item.breed.name);
    return { type: 'mixed', label: `혼합 · ${names.join(' + ')}`, matches };
  }

  buildDiffusionPaths() {
    const steps = parseInt(this.diffusionStepSelect.value, 10);
    const dt = 1 / steps;
    const center = { x: this.canvas.width * 0.12, y: this.canvas.height * 0.50 };
    this.diffusionSteps = steps;
    this.diffusionStepComponents = [];
    this.diffusionPaths = this.noiseGrid.map((cell, idx) => {
      const rng = mulberry32(this.seed * 100003 + idx * 7919 + this.diffusionNoiseRun * 104729 + 17);
      let pos = { ...cell.z0 };
      const path = [pos];
      const components = [];

      for (let step = 0; step < steps; step++) {
        const tau = step / steps;
        const stats = this.getDiffusionStats(pos, tau);
        const driftX = 0.5 * stats.beta * stats.y.x + stats.beta * stats.score.x;
        const driftY = 0.5 * stats.beta * stats.y.y + stats.beta * stats.score.y;
        const [noiseX, noiseY] = normalPair(rng);
        const diffusionScale = Math.sqrt(stats.beta * dt);
        const drift = { x: driftX * dt, y: driftY * dt };
        const noise = { x: diffusionScale * noiseX, y: diffusionScale * noiseY };
        const nextY = {
          x: stats.y.x + drift.x + noise.x,
          y: stats.y.y + drift.y + noise.y
        };
        const nextPos = {
          x: center.x + nextY.x * stats.scaleX,
          y: center.y + nextY.y * stats.scaleY
        };
        components.push({
          from: pos,
          driftEnd: {
            x: pos.x + drift.x * stats.scaleX,
            y: pos.y + drift.y * stats.scaleY
          },
          to: nextPos,
          drift,
          noise,
          total: { x: drift.x + noise.x, y: drift.y + noise.y }
        });
        pos = nextPos;
        path.push(pos);
      }
      this.diffusionStepComponents[idx] = components;
      return path;
    });
  }

  getActivePredictedBreed(t) {
    const cell = this.noiseGrid[this.selectedCellIndex] || this.noiseGrid[0];
    if (this.algorithm === 'flow') return cell;
    const pos = this.computeTrajectoryPos(cell.z0, cell.x1, t, this.selectedCellIndex);
    return this.getDiffusionStats(pos, t).breed;
  }

  getRealtimeParticleTarget(cell, t, idx) {
    if (this.algorithm === 'flow') return cell;
    const pos = this.computeTrajectoryPos(cell.z0, cell.x1, t, idx);
    return this.getDiffusionStats(pos, t).breed;
  }

  getCombinedCenterPoint(t) {
    let sumX = 0;
    let sumY = 0;
    let totalWeight = 0;

    for (let i = 0; i < this.noiseGrid.length; i++) {
      const cell = this.noiseGrid[i];
      const pos = this.computeTrajectoryPos(cell.z0, cell.x1, t, i);
      sumX += pos.x;
      sumY += pos.y;
      totalWeight += 1;
    }

    return { x: sumX / totalWeight, y: sumY / totalWeight };
  }

  getPriorBounds() {
    const center = { x: this.canvas.width * 0.12, y: this.canvas.height * 0.50 };
    const halfSize = Math.min(this.canvas.width * 0.095, this.canvas.height * 0.17);
    return {
      x: center.x - halfSize,
      y: center.y - halfSize,
      width: halfSize * 2,
      height: halfSize * 2
    };
  }

  getVelocityVector(cell, t, idx) {
    const pos = this.computeTrajectoryPos(cell.z0, cell.x1, t, idx);

    if (this.algorithm === 'flow') {
      return {
        vx: (cell.x1.x - cell.z0.x) / this.noiseScaleX,
        vy: (cell.x1.y - cell.z0.y) / this.noiseScaleY,
        isConstant: true
      };
    } else {
      const stats = this.getDiffusionStats(pos, t);
      return {
        vx: 0.5 * stats.beta * stats.y.x + stats.beta * stats.score.x,
        vy: 0.5 * stats.beta * stats.y.y + stats.beta * stats.score.y,
        isConstant: false
      };
    }
  }

  getCurrentRgbLatent(cell, pos) {
    const center = { x: this.canvas.width * 0.12, y: this.canvas.height * 0.50 };
    const x = (pos.x - center.x) / this.noiseScaleX;
    const y = (pos.y - center.y) / this.noiseScaleY;
    const mean = (cell.rgbNoise.r + cell.rgbNoise.g + cell.rgbNoise.b) / 3;
    return {
      r: x / Math.sqrt(2) + y / Math.sqrt(6) + mean,
      g: -x / Math.sqrt(2) + y / Math.sqrt(6) + mean,
      b: -2 * y / Math.sqrt(6) + mean
    };
  }

  renderModeResponsibilities(responsibilities, outsideWeight = 0) {
    const knownWeight = 1 - outsideWeight;
    const top = responsibilities.slice(0, outsideWeight > 0 ? 2 : 3)
      .map(item => ({ ...item, responsibility: item.responsibility * knownWeight }));
    const remainingKnown = Math.max(0, knownWeight - top.reduce((sum, item) => sum + item.responsibility, 0));
    const rows = [
      ...top,
      {
        breed: { name: `나머지 ${Math.max(0, responsibilities.length - top.length)}개 known mode`, shortName: '나머지', color: '#64748b' },
        responsibility: remainingKnown
      }
    ];
    if (outsideWeight > 0) {
      rows.push({
        breed: { name: '2.5σ 경계 밖', shortName: '기타', color: '#fb7185' },
        responsibility: outsideWeight
      });
    }
    rows.sort((a, b) => b.responsibility - a.responsibility);

    this.modeResponsibilityBars.innerHTML = rows.map((item, index) => {
      const percentage = item.responsibility * 100;
      const name = item.breed.shortName || item.breed.name;
      return `
        <div class="mode-candidate-row" title="${item.breed.name}">
          <span class="mode-candidate-name">${index === 0 ? '🔥 ' : ''}${name}</span>
          <span class="mode-candidate-track"><span class="mode-candidate-fill" style="display:block;width:${percentage}%;background:${item.breed.color}"></span></span>
          <span class="mode-candidate-value">${percentage.toFixed(1)}%</span>
        </div>
      `;
    }).join('');
  }

  updateSelectedCellDetail() {
    const cell = this.noiseGrid[this.selectedCellIndex];
    if (!cell) return;

    const activeBreed = this.getActivePredictedBreed(this.currentTime);
    const velVector = this.getVelocityVector(cell, this.currentTime, this.selectedCellIndex);
    this.winningBreed = activeBreed;

    const isDiff = this.algorithm === 'diff';
    const pos = this.computeTrajectoryPos(cell.z0, cell.x1, this.currentTime, this.selectedCellIndex);
    const rgbZt = this.getCurrentRgbLatent(cell, pos);
    const stepComponent = this.getDiffusionStepComponent(this.currentTime, this.selectedCellIndex);
    const diffusionStats = isDiff ? this.getDiffusionStats(pos, this.currentTime) : null;
    const finalStatus = isDiff && this.currentTime >= 0.999 ? this.getFinalModeStatus(pos) : null;
    const confidence = isDiff ? Math.round(diffusionStats.responsibility * 100) : 100;
    const mainShortName = activeBreed.shortName || activeBreed.breedName;

    this.selectedCellName.textContent = `z₀-${cell.name} (시드 #${this.seed})`;
    this.selectedCellVal.textContent = `RGB z₀ = [${cell.rgbNoise.r >= 0 ? '+' : ''}${cell.rgbNoise.r.toFixed(2)}, ${cell.rgbNoise.g >= 0 ? '+' : ''}${cell.rgbNoise.g.toFixed(2)}, ${cell.rgbNoise.b >= 0 ? '+' : ''}${cell.rgbNoise.b.toFixed(2)}]`;
    this.selectedCellProjection.textContent = `2D 투영 π(z₀) = [${cell.noiseValue.x >= 0 ? '+' : ''}${cell.noiseValue.x.toFixed(2)}, ${cell.noiseValue.y >= 0 ? '+' : ''}${cell.noiseValue.y.toFixed(2)}]`;
    this.selectedCellTarget.textContent = isDiff
      ? (finalStatus ? finalStatus.label : `${mainShortName} (가장 큰 영향 mode)`)
      : `${cell.breedName} (조건부 pairing)`;

    if (this.metricRgbZt) this.metricRgbZt.textContent = `[${rgbZt.r >= 0 ? '+' : ''}${rgbZt.r.toFixed(2)}, ${rgbZt.g >= 0 ? '+' : ''}${rgbZt.g.toFixed(2)}, ${rgbZt.b >= 0 ? '+' : ''}${rgbZt.b.toFixed(2)}]`;
    if (this.metricVectorVal) this.metricVectorVal.textContent = `[x: ${velVector.vx > 0 ? '+' : ''}${velVector.vx.toFixed(1)}, y: ${velVector.vy > 0 ? '+' : ''}${velVector.vy.toFixed(1)}]`;
    if (this.metricVectorLabel) this.metricVectorLabel.innerHTML = isDiff ? 'reverse-SDE drift b<sub>τ</sub>:' : '조건부 목표 u<sub>τ</sub>:';
    if (this.metricVectorStatus) this.metricVectorStatus.textContent = isDiff ? 'score + VP drift (시간에 따라 변화)' : '선택한 한 쌍에서만 일정';
    this.diffusionStepBreakdown.hidden = !isDiff;
    if (isDiff && stepComponent) {
      const formatStep = vector => `[${vector.x >= 0 ? '+' : ''}${vector.x.toFixed(2)}, ${vector.y >= 0 ? '+' : ''}${vector.y.toFixed(2)}]`;
      this.metricDriftStep.textContent = formatStep(stepComponent.drift);
      this.metricNoiseStep.textContent = formatStep(stepComponent.noise);
      this.metricTotalStep.textContent = formatStep(stepComponent.total);
    }
    if (this.metricNearestName) this.metricNearestName.textContent = `${mainShortName}`;
    this.modeBoundaryRow.hidden = !isDiff;
    if (isDiff) {
      this.modeBoundaryStatus.textContent = finalStatus ? finalStatus.label : '진행 중 · τ=1에서 판정';
      this.modeBoundaryStatus.style.color = !finalStatus ? '#94a3b8'
        : finalStatus.type === 'single' ? '#22c55e'
        : finalStatus.type === 'mixed' ? '#facc15' : '#fb7185';
    }
    if (this.metricConfidenceLabel) this.metricConfidenceLabel.textContent = isDiff ? 'mode responsibility:' : '조건부 pairing:';
    if (this.metricConfidence) this.metricConfidence.textContent = `${confidence}%`;
    if (this.metricConfBar) this.metricConfBar.style.width = `${confidence}%`;
    this.metricConfidenceRow.hidden = isDiff;
    this.metricConfidenceBar.hidden = isDiff;
    this.modeResponsibilityList.hidden = !isDiff;
    if (isDiff) this.renderModeResponsibilities(diffusionStats.modeResponsibilities, finalStatus?.outsideWeight || 0);
    if (this.targetBreedBadge) {
      this.targetBreedBadge.textContent = isDiff
        ? (finalStatus ? `최종 판정: ${finalStatus.label}` : `τ=${this.currentTime.toFixed(2)} 가장 큰 영향: ${mainShortName}`)
        : `조건부 목적지: ${mainShortName}`;
    }
  }

  renderEvolutionCanvases() {
    [0, 5, 10].forEach(stepIdx => {
      const t = stepIdx / 10;
      const evoCanvas = document.getElementById(`evo-canvas-${stepIdx}`);
      if (!evoCanvas) return;
      const ectx = evoCanvas.getContext('2d');
      const w = evoCanvas.width;
      const h = evoCanvas.height;
      ectx.clearRect(0, 0, w, h);

      const cell = this.noiseGrid[this.selectedCellIndex] || this.noiseGrid[0];
      if (!cell) return;
      if (this.algorithm === 'diff') {
        const pos = this.computeTrajectoryPos(cell.z0, cell.x1, t, this.selectedCellIndex);
        const stats = this.getDiffusionStats(pos, t);
        const status = t >= 0.999 ? this.getFinalModeStatus(pos) : null;
        if (status && status.type !== 'single') this.drawUnresolvedMode(ectx, w, h, stats, status);
        else this.drawDenoisedCatOnContext(ectx, w, h, status?.breed || stats.breed, t);
      } else {
        this.drawDenoisedCatOnContext(ectx, w, h, cell, t);
      }
    });
  }

  drawUnresolvedMode(ctx, w, h, stats, status) {
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, w, h);
    const knownWeight = 1 - (status.outsideWeight || 0);
    stats.modeResponsibilities.slice(0, 3).forEach((item, index) => {
      ctx.save();
      ctx.globalAlpha = 0.2 + item.responsibility * knownWeight * 0.8;
      ctx.fillStyle = item.breed.color;
      ctx.beginPath();
      ctx.arc(w / 2 + (index - 1) * w * 0.12, h / 2, Math.min(w, h) * 0.22, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
    ctx.fillStyle = '#ffffff';
    ctx.font = `700 ${w > 100 ? 16 : 11}px Outfit`;
    ctx.textAlign = 'center';
    const unresolvedLabel = status.type === 'outside'
      ? `기타 ${Math.round(status.outsideWeight * 100)}%`
      : '혼합';
    ctx.fillText(unresolvedLabel, w / 2, h / 2 + 5);
  }

  drawDenoisedCatOnContext(ctx, w, h, breedData, t) {
    const isNoisePhase = t < 0.05;

    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, w, h);

    if (isNoisePhase) {
      const imgData = ctx.createImageData(w, h);
      const data = imgData.data;
      const rng = mulberry32(this.seed * 65537 + (breedData.id || this.selectedCellIndex) * 257);
      for (let i = 0; i < data.length; i += 4) {
        const [noiseR, noiseG] = normalPair(rng);
        const [noiseB] = normalPair(rng);
        data[i] = Math.max(0, Math.min(255, 128 + noiseR * 52));
        data[i+1] = Math.max(0, Math.min(255, 128 + noiseG * 52));
        data[i+2] = Math.max(0, Math.min(255, 128 + noiseB * 52));
        data[i+3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
      return;
    }

    ctx.save();
    ctx.translate(w / 2, h / 2 - (w > 100 ? 10 : 2));

    const mainColor = breedData.targetColor;

    const catOpacity = Math.min(1.0, Math.max(0.15, Math.pow(t, 0.7)));
    ctx.globalAlpha = catOpacity;

    const scale = w / 200;
    ctx.scale(scale, scale);

    if (this.prompt === 'cat') {
      ctx.fillStyle = mainColor;
      ctx.beginPath(); ctx.arc(0, 10, 48, 0, Math.PI * 2); ctx.fill();

      if (t >= 0.25) {
        const isFolded = breedData.pattern === 'folded';
        ctx.beginPath();
        if (isFolded) {
          ctx.ellipse(-30, -25, 18, 12, -0.4, 0, Math.PI*2);
          ctx.ellipse(30, -25, 18, 12, 0.4, 0, Math.PI*2);
        } else {
          ctx.moveTo(-40, -15); ctx.lineTo(-60, -65); ctx.lineTo(-12, -40);
          ctx.moveTo(40, -15); ctx.lineTo(60, -65); ctx.lineTo(12, -40);
        }
        ctx.fill();
      }

      if (t >= 0.45) {
        if (breedData.pattern === 'mask') {
          ctx.fillStyle = '#1e1b18';
          ctx.beginPath(); ctx.ellipse(0, 10, 30, 22, 0, 0, Math.PI*2); ctx.fill();
        } else if (breedData.pattern === 'stripes') {
          ctx.strokeStyle = '#b45309';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(-15, -20); ctx.lineTo(-8, -4);
          ctx.moveTo(0, -25); ctx.lineTo(0, -8);
          ctx.moveTo(15, -20); ctx.lineTo(8, -4);
          ctx.stroke();
        }
      }

      if (t >= 0.60) {
        ctx.fillStyle = '#060911';
        ctx.beginPath();
        ctx.ellipse(-18, 0, 9, 6, 0, 0, Math.PI * 2);
        ctx.ellipse(18, 0, 9, 6, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = breedData.pattern === 'blue' ? '#38bdf8' : '#facc15';
        ctx.beginPath();
        ctx.arc(-18, 0, 3.5, 0, Math.PI * 2);
        ctx.arc(18, 0, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }

    } else if (this.prompt === 'dog') {
      ctx.fillStyle = mainColor;

      if (t >= 0.15) {
        if (breedData.pattern === 'shiba') {
          ctx.beginPath(); ctx.arc(0, 0, 48, 0, Math.PI*2); ctx.fill();
          ctx.beginPath(); ctx.moveTo(-35, -20); ctx.lineTo(-55, -60); ctx.lineTo(-10, -40); ctx.fill();
          ctx.beginPath(); ctx.moveTo(35, -20); ctx.lineTo(55, -60); ctx.lineTo(10, -40); ctx.fill();
          ctx.fillStyle = '#ffffff';
          ctx.beginPath(); ctx.ellipse(0, 18, 22, 16, 0, 0, Math.PI*2); ctx.fill();
        } else {
          ctx.beginPath(); ctx.arc(0, 0, 48, 0, Math.PI * 2); ctx.fill();
          ctx.beginPath();
          ctx.ellipse(-45, 8, 18, 35, 0.3, 0, Math.PI * 2);
          ctx.ellipse(45, 8, 18, 35, -0.3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (t >= 0.60) {
        ctx.fillStyle = breedData.pattern === 'husky' ? '#38bdf8' : '#1e1b18';
        ctx.beginPath();
        ctx.arc(-16, -5, 5, 0, Math.PI*2);
        ctx.arc(16, -5, 5, 0, Math.PI*2);
        ctx.fill();
        ctx.fillStyle = '#090d16';
        ctx.beginPath(); ctx.ellipse(0, 12, 8, 6, 0, 0, Math.PI * 2); ctx.fill();
      }

    }

    ctx.restore();

    if (t < 0.95) {
      const noiseAlpha = (1 - t) * 0.75;
      const overlayData = ctx.getImageData(0, 0, w, h);
      const pix = overlayData.data;
      const rng = mulberry32(this.seed * 8191 + Math.floor(t * 100) * 131 + (breedData.id || this.selectedCellIndex));
      for (let i = 0; i < pix.length; i += 4) {
        if (rng() < noiseAlpha) {
          const [gaussian] = normalPair(rng);
          const grain = gaussian * 45;
          pix[i] = Math.max(0, Math.min(255, pix[i] + grain));
          pix[i+1] = Math.max(0, Math.min(255, pix[i+1] + grain));
          pix[i+2] = Math.max(0, Math.min(255, pix[i+2] + grain));
        }
      }
      ctx.putImageData(overlayData, 0, 0);
    }
  }

  renderTargetImage() {
    const cell = this.noiseGrid[this.selectedCellIndex];
    if (!cell) return;

    const ctx = this.targetCtx;
    const w = this.targetCanvas.width;
    const h = this.targetCanvas.height;
    const isDiff = this.algorithm === 'diff';
    const pos = this.computeTrajectoryPos(cell.z0, cell.x1, this.currentTime, this.selectedCellIndex);
    const stats = isDiff ? this.getDiffusionStats(pos, this.currentTime) : null;
    const finalStatus = isDiff && this.currentTime >= 0.999 ? this.getFinalModeStatus(pos) : null;
    const breedData = isDiff ? (finalStatus?.breed || stats.breed) : cell;

    ctx.clearRect(0, 0, w, h);

    if (finalStatus && finalStatus.type !== 'single') this.drawUnresolvedMode(ctx, w, h, stats, finalStatus);
    else this.drawDenoisedCatOnContext(ctx, w, h, breedData, this.currentTime);

    ctx.fillStyle = '#ffffff';
    ctx.font = '700 12px Outfit';
    ctx.textAlign = 'center';

    const mainShortName = breedData.shortName || breedData.breedName;
    const label = finalStatus
      ? `최종 판정: ${finalStatus.label}`
      : `${isDiff ? `τ=${this.currentTime.toFixed(2)} 가장 큰 영향: ` : '조건부 목적지: '}${mainShortName}`;
    ctx.fillText(label, w / 2, h - 10);

    this.renderBatchMeanImage();
    this.renderEvolutionCanvases();
  }

  renderBatchMeanImage() {
    const isDiff = this.algorithm === 'diff';
    this.batchMeanPreview.hidden = !isDiff;
    if (!isDiff) return;

    const pos = this.getCombinedCenterPoint(this.currentTime);
    const stats = this.getDiffusionStats(pos, this.currentTime);
    const status = this.currentTime >= 0.999 ? this.getFinalModeStatus(pos) : null;
    const breed = status?.breed || stats.breed;
    const ctx = this.batchMeanCanvas.getContext('2d');
    const { width, height } = this.batchMeanCanvas;

    ctx.clearRect(0, 0, width, height);
    if (status && status.type !== 'single') this.drawUnresolvedMode(ctx, width, height, stats, status);
    else this.drawDenoisedCatOnContext(ctx, width, height, breed, this.currentTime);
    this.batchMeanBadge.textContent = status
      ? `batch mean 최종 판정: ${status.label}`
      : `τ=${this.currentTime.toFixed(2)} · batch mean 가장 큰 영향: ${breed.shortName || breed.name}`;
  }

  loop(timestamp) {
    const dt = (timestamp - this.lastTimestamp) / 1000;
    this.lastTimestamp = timestamp;

    if (this.isPlaying) {
      this.currentTime += (dt * 0.18) * this.speed;
      if (this.currentTime > 1.0) {
        if (this.isRecording) this.stopRecording();
        this.currentTime = 0;
      }
      this.scrubber.value = Math.floor(this.currentTime * 1000);
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    }

    if (this.isRecording) {
      const elapsed = ((performance.now() - this.recStartTime) / 1000).toFixed(1);
      this.recTimer.textContent = elapsed + 's';
    }

    this.renderManifold();
    requestAnimationFrame((ts) => this.loop(ts));
  }

  drawArrowHead(ctx, fromX, fromY, toX, toY, color, width) {
    const headlen = 9;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;

    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headlen * Math.cos(angle - Math.PI / 6), toY - headlen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(toX - headlen * Math.cos(angle + Math.PI / 6), toY - headlen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  drawDiffusionHistory(ctx, idx, t, color, width) {
    const path = this.diffusionPaths[idx];
    const completedStep = Math.floor(t * this.diffusionSteps);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(path[0].x, path[0].y);
    for (let step = 1; step <= completedStep; step++) ctx.lineTo(path[step].x, path[step].y);
    ctx.stroke();
    ctx.restore();
  }

  drawDiffusionStepDecomposition(ctx, idx, t) {
    const component = this.getDiffusionStepComponent(t, idx);
    if (!component) return;

    const drift = {
      x: component.driftEnd.x - component.from.x,
      y: component.driftEnd.y - component.from.y
    };
    const noise = {
      x: component.to.x - component.driftEnd.x,
      y: component.to.y - component.driftEnd.y
    };
    const total = { x: drift.x + noise.x, y: drift.y + noise.y };
    const maxLength = Math.max(Math.hypot(drift.x, drift.y), Math.hypot(noise.x, noise.y), Math.hypot(total.x, total.y), 1);
    const scale = Math.min(1, 90 / maxLength);
    const base = component.from;
    const driftEnd = { x: base.x + drift.x * scale, y: base.y + drift.y * scale };
    const totalEnd = { x: base.x + total.x * scale, y: base.y + total.y * scale };

    this.drawArrowHead(ctx, base.x, base.y, totalEnd.x, totalEnd.y, '#f8fafc', 5);
    this.drawArrowHead(ctx, base.x, base.y, driftEnd.x, driftEnd.y, '#facc15', 3);
    this.drawArrowHead(ctx, driftEnd.x, driftEnd.y, totalEnd.x, totalEnd.y, '#fb7185', 3);

    ctx.save();
    ctx.font = '700 10px JetBrains Mono';
    ctx.fillStyle = '#facc15';
    ctx.fillText('drift', driftEnd.x + 4, driftEnd.y - 5);
    ctx.fillStyle = '#fb7185';
    ctx.fillText('noise', totalEnd.x + 4, totalEnd.y + 12);
    ctx.fillStyle = '#f8fafc';
    ctx.fillText('Δz', totalEnd.x + 4, totalEnd.y - 5);
    ctx.restore();
  }

  drawModeCandidateRays(ctx, pos, stats) {
    stats.modeResponsibilities.slice(0, 3).forEach(({ breed, responsibility }) => {
      ctx.save();
      ctx.globalAlpha = Math.min(0.7, 0.15 + responsibility);
      ctx.strokeStyle = breed.color;
      ctx.lineWidth = 1 + responsibility * 7;
      ctx.setLineDash([2, 6]);
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
      ctx.lineTo(breed.x1.x, breed.x1.y);
      ctx.stroke();
      ctx.restore();
    });
  }

  // RENDER DYNAMIC BREED CLUSTERS ON 2D MANIFOLD!
  // In Diffusion SDE mode, renders active target breed clusters dynamically across ALL 20 SPECIES LOCATIONS!
  renderManifold() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    ctx.fillStyle = '#040711';
    ctx.fillRect(0, 0, w, h);

    // 1. Entire Model Data Distribution p(x)
    ctx.save();
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.35)';
    ctx.fillStyle = 'rgba(99, 102, 241, 0.05)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(w * 0.5, h * 0.55, w * 0.46, h * 0.43, -0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#a5b4fc';
    ctx.font = '600 13px Outfit';
    ctx.fillText('2D toy 데이터 공간 [개, 고양이]', w * 0.15, h * 0.20);
    ctx.restore();

    // 2. Prompt Class Distributions
    const promptConfigs = [
      { id: 'cat', label: '고양이 분포 p(x|"고양이")', center: { x: w * 0.65, y: h * 0.45 }, rx: w * 0.175, ry: h * 0.250, color: '#38bdf8' },
      { id: 'dog', label: '강아지 분포 p(x|"강아지")', center: { x: w * 0.62, y: h * 0.72 }, rx: w * 0.162, ry: h * 0.226, color: '#f59e0b' }
    ];

    promptConfigs.forEach(cfg => {
      const isSelected = cfg.id === this.prompt;
      ctx.save();
      ctx.strokeStyle = isSelected ? cfg.color : 'rgba(255, 255, 255, 0.12)';
      ctx.fillStyle = isSelected ? `${cfg.color}15` : 'rgba(255, 255, 255, 0.02)';
      ctx.lineWidth = isSelected ? 2.5 : 1;

      ctx.beginPath();
      ctx.ellipse(cfg.center.x, cfg.center.y, cfg.rx, cfg.ry, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      if (isSelected) {
        ctx.fillStyle = cfg.color;
        ctx.font = '700 13px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText(`★ ${cfg.label}`, cfg.center.x, cfg.center.y - cfg.ry - 16);
      }
      ctx.restore();
    });

    // Calculate Dynamic Combined Batch Centroid (\bar{x}_{center})
    const centerPt = this.getCombinedCenterPoint(this.currentTime);
    let activeBreed = this.getActivePredictedBreed(this.currentTime);
    if (this.algorithm === 'diff' && this.currentTime >= 0.999) {
      const cell = this.noiseGrid[this.selectedCellIndex];
      const status = this.getFinalModeStatus(this.getDiffusionPosition(this.currentTime, this.selectedCellIndex));
      activeBreed = status.type === 'single' ? status.breed : null;
    }

    // 3. Breed Sub-Clusters on Manifold
    const activeClustersToDraw = this.algorithm === 'diff' ? this.allBreedFixedClusters : this.fixedBreedMap;

    if (activeClustersToDraw) {
      const boundaryRadius = this.dataModeSigma * this.modeBoundarySigma * this.noiseScaleX;
      activeClustersToDraw.forEach((breedCluster) => {
        const isWinner = activeBreed && (activeBreed.breedName === breedCluster.name || activeBreed.name === breedCluster.name);
        
        ctx.save();
        ctx.strokeStyle = isWinner ? '#ec4899' : `${breedCluster.color}80`;
        ctx.fillStyle = isWinner ? 'rgba(236, 72, 153, 0.35)' : `${breedCluster.color}20`;
        ctx.lineWidth = isWinner ? 3.0 : 1.0;

        ctx.beginPath();
        ctx.arc(breedCluster.x1.x, breedCluster.x1.y, boundaryRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        if (isWinner) {
          ctx.shadowBlur = 15;
          ctx.shadowColor = '#ec4899';
          ctx.stroke();
        }

        ctx.fillStyle = isWinner ? '#ffffff' : '#9ca3af';
        ctx.font = isWinner ? '700 11px Outfit' : '500 9px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText(`${isWinner ? '🔥 ' : ''}${breedCluster.shortName}`, breedCluster.x1.x, breedCluster.x1.y - boundaryRadius - 5);
        ctx.restore();
      });
    }

    // 4. Initial Noise Points z0 on Left Side
    const priorBounds = this.getPriorBounds();
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.rect(priorBounds.x, priorBounds.y, priorBounds.width, priorBounds.height);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 12px JetBrains Mono';
    ctx.fillText(`Gaussian prior z₀ (시드 #${this.seed})`, priorBounds.x, priorBounds.y - 12);
    ctx.restore();

    // 5. Render the selected sample's path for the active metric mode.
    this.noiseGrid.forEach((cell, idx) => {
      const isSelectedCell = idx === this.selectedCellIndex;
      const isFlow = this.algorithm === 'flow';
      const pos = isFlow
        ? this.getFlowPosition(cell, this.currentTime)
        : this.getDiffusionPosition(this.currentTime, idx);
      const velVec = this.getVelocityVector(cell, this.currentTime, idx);

      if (isSelectedCell && !isFlow) {
        this.drawModeCandidateRays(ctx, pos, this.getDiffusionStats(pos, this.currentTime));
      }

      if (isSelectedCell) {
        if (isFlow) {
          ctx.save();
          ctx.strokeStyle = '#00f0ff';
          ctx.lineWidth = 3;
          ctx.shadowBlur = 10;
          ctx.shadowColor = '#00f0ff';
          ctx.beginPath();
          ctx.moveTo(cell.z0.x, cell.z0.y);
          ctx.lineTo(pos.x, pos.y);
          ctx.stroke();
          ctx.restore();
        } else {
          this.drawDiffusionHistory(ctx, idx, this.currentTime, '#ff007f', 3);
        }
      }

      // Initial Noise Dot z0
      ctx.save();
      ctx.fillStyle = `rgb(${cell.r}, ${cell.g}, ${cell.b})`;
      ctx.beginPath();
      ctx.arc(cell.z0.x, cell.z0.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.fillStyle = isFlow ? '#38bdf8' : '#f43f5e';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isSelectedCell ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fill();
      if (isSelectedCell) ctx.stroke();
      ctx.restore();

      if (isSelectedCell && this.currentTime < 0.98) {
        if (isFlow) {
          const vecNorm = Math.sqrt(velVec.vx * velVec.vx + velVec.vy * velVec.vy) || 1;
          const endX = pos.x + (velVec.vx / vecNorm) * 48;
          const endY = pos.y + (velVec.vy / vecNorm) * 48;
          this.drawArrowHead(ctx, pos.x, pos.y, endX, endY, '#00f0ff', 3.2);
          const realtimeTarget = this.getRealtimeParticleTarget(cell, this.currentTime, idx);
          ctx.save();
          ctx.fillStyle = '#00f0ff';
          ctx.font = '700 11px JetBrains Mono';
          ctx.shadowBlur = 8;
          ctx.shadowColor = '#00f0ff';
          ctx.fillText(`u_${realtimeTarget.name} [${velVectorStr(velVec.vx)}, ${velVectorStr(velVec.vy)}]`, endX + 4, endY - 6);
          ctx.restore();
        } else {
          this.drawDiffusionStepDecomposition(ctx, idx, this.currentTime);
        }
      }
    });

    function velVectorStr(val) {
      return (val > 0 ? '+' : '') + val.toFixed(1);
    }

    // 6. RENDER DYNAMIC COMBINED BATCH CENTROID (\bar{x}_{center}) & RAY TO ACTIVE WINNER
    if (centerPt) {
      ctx.save();
      
      if (activeBreed) {
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2.0;
        ctx.setLineDash([3, 3]);
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#facc15';

        ctx.beginPath();
        ctx.moveTo(centerPt.x, centerPt.y);
        ctx.lineTo(activeBreed.x1.x, activeBreed.x1.y);
        ctx.stroke();
      }

      ctx.fillStyle = '#facc15';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.shadowBlur = 18;
      ctx.shadowColor = '#facc15';

      ctx.beginPath();
      ctx.arc(centerPt.x, centerPt.y, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = '700 12px Outfit';
      ctx.fillText('★ batch mean (x̄)', centerPt.x - 40, centerPt.y - 14);

      ctx.restore();
    }
  }

  getFlowPosition(cell, t) {
    return {
      x: (1 - t) * cell.z0.x + t * cell.x1.x,
      y: (1 - t) * cell.z0.y + t * cell.x1.y
    };
  }

  getDiffusionPosition(t, seedOffset) {
    const path = this.diffusionPaths[seedOffset];
    const scaledStep = Math.max(0, Math.min(this.diffusionSteps, t * this.diffusionSteps));
    const step = Math.floor(scaledStep);
    const nextStep = Math.min(this.diffusionSteps, step + 1);
    const fraction = scaledStep - step;
    return {
      x: path[step].x * (1 - fraction) + path[nextStep].x * fraction,
      y: path[step].y * (1 - fraction) + path[nextStep].y * fraction
    };
  }

  getDiffusionStepComponent(t, seedOffset) {
    const components = this.diffusionStepComponents[seedOffset];
    const step = Math.min(this.diffusionSteps - 1, Math.floor(t * this.diffusionSteps));
    return components && components[Math.max(0, step)];
  }

  computeTrajectoryPos(z0, x1, t, seedOffset) {
    if (this.algorithm === 'flow') return this.getFlowPosition({ z0, x1 }, t);
    return this.getDiffusionPosition(t, seedOffset);
  }

  selfCheck() {
    const path = this.diffusionPaths[0];
    const components = this.diffusionStepComponents[0];
    const firstComponent = components[0];
    const stats = this.getDiffusionStats(path[Math.floor(path.length / 2)], 0.5);
    const responsibilitySum = stats.modeResponsibilities.reduce((sum, item) => sum + item.responsibility, 0);
    const cell = this.noiseGrid[0];
    const rgbAtStart = this.getCurrentRgbLatent(cell, cell.z0);
    const projectedX = (cell.rgbNoise.r - cell.rgbNoise.g) / Math.sqrt(2);
    const projectedY = (cell.rgbNoise.r + cell.rgbNoise.g - 2 * cell.rgbNoise.b) / Math.sqrt(6);
    const priorBounds = this.getPriorBounds();
    const allNoiseInsidePrior = this.noiseGrid.every(noise =>
      noise.z0.x >= priorBounds.x &&
      noise.z0.x <= priorBounds.x + priorBounds.width &&
      noise.z0.y >= priorBounds.y &&
      noise.z0.y <= priorBounds.y + priorBounds.height
    );
    const outsideStatus = this.getFinalModeStatus({ x: -1e6, y: -1e6 });
    const expectedNoiseScale = Math.min(priorBounds.width, priorBounds.height) * 0.18;
    if (
      path.length !== this.diffusionSteps + 1 ||
      components.length !== this.diffusionSteps ||
      Math.abs(firstComponent.drift.x + firstComponent.noise.x - firstComponent.total.x) > 1e-12 ||
      Math.abs(firstComponent.drift.y + firstComponent.noise.y - firstComponent.total.y) > 1e-12 ||
      Math.abs(firstComponent.to.x - path[1].x) > 1e-12 ||
      Math.abs(firstComponent.to.y - path[1].y) > 1e-12 ||
      !Number.isFinite(stats.score.x) ||
      !Number.isFinite(stats.score.y) ||
      Math.abs(responsibilitySum - 1) > 1e-12 ||
      stats.responsibility <= 0 ||
      stats.responsibility > 1 ||
      Math.abs(cell.noiseValue.x - projectedX) > 1e-12 ||
      Math.abs(cell.noiseValue.y - projectedY) > 1e-12 ||
      Math.abs(cell.rgbNoise.r - rgbAtStart.r) > 1e-12 ||
      Math.abs(cell.rgbNoise.g - rgbAtStart.g) > 1e-12 ||
      Math.abs(cell.rgbNoise.b - rgbAtStart.b) > 1e-12 ||
      !allNoiseInsidePrior ||
      this.noiseScaleX !== this.noiseScaleY ||
      Math.abs(this.noiseScaleX - expectedNoiseScale) > 1e-12 ||
      outsideStatus.type !== 'outside' ||
      outsideStatus.outsideWeight !== 0.85
    ) {
      throw new Error('Diffusion toy model self-check failed');
    }
  }

  toggleRecording() {
    if (this.isRecording) this.stopRecording();
    else this.startRecording();
  }

  startRecording() {
    this.recordedChunks = [];
    const stream = this.canvas.captureStream(60);
    let options = { mimeType: 'video/webm;codecs=vp9' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) options = { mimeType: 'video/webm' };

    try {
      this.mediaRecorder = new MediaRecorder(stream, options);
    } catch (e) {
      alert('비디오 녹화를 지원하지 않는 브라우저입니다.');
      return;
    }

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.recordedChunks.push(e.data);
    };

    this.mediaRecorder.onstop = () => {
      const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `generative_explainer_${this.prompt}_${Date.now()}.webm`;
      a.click();
      URL.revokeObjectURL(url);
    };

    this.mediaRecorder.start();
    this.isRecording = true;
    this.recBadge.classList.remove('hidden');
    this.recStartTime = performance.now();
    this.btnExportVideo.innerHTML = '<i data-lucide="square"></i> 녹화 중지 & 저장';
    if (window.lucide) { lucide.createIcons(); }

    this.currentTime = 0;
    this.isPlaying = true;
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      this.recBadge.classList.add('hidden');
      this.btnExportVideo.innerHTML = '<i data-lucide="video"></i> HD 영상 다운로드 (WebM)';
      if (window.lucide) { lucide.createIcons(); }
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.explainer = new ExplainerStudio();
});
