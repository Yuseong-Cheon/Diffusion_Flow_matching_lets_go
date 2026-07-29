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
    this.metricConfidence = document.getElementById('metric-confidence');
    this.metricConfidenceLabel = document.getElementById('metric-confidence-label');
    this.metricConfBar = document.getElementById('metric-conf-bar');

    this.btnModeFlow = document.getElementById('btn-mode-flow');
    this.btnModeDiff = document.getElementById('btn-mode-diff');

    this.btnPlayPause = document.getElementById('btn-play-pause');
    this.btnReset = document.getElementById('btn-reset');
    this.scrubber = document.getElementById('scrubber');
    this.diffusionStepSelect = document.getElementById('diffusion-step-select');
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
      this.generateSeedData();
    });

    this.seedSlider.addEventListener('input', (e) => {
      this.seed = parseInt(e.target.value);
      this.seedBadge.textContent = `시드 #${this.seed}`;
      this.generateSeedData();
    });

    this.btnRandomSeed.addEventListener('click', () => {
      this.seed = Math.floor(Math.random() * 999) + 1;
      this.seedSlider.value = this.seed;
      this.seedBadge.textContent = `시드 #${this.seed}`;
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

    for (let i = 0; i < 9; i++) {
      const [noiseR, noiseG] = normalPair(rng);
      const [noiseB] = normalPair(rng);
      const noiseX = (noiseR - noiseG) / Math.sqrt(2);
      const noiseY = (noiseR + noiseG - 2 * noiseB) / Math.sqrt(6);
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

    const priorBounds = this.getPriorBounds();
    const center = { x: priorBounds.x + priorBounds.width / 2, y: priorBounds.y + priorBounds.height / 2 };
    const maxNoiseX = Math.max(1, ...this.noiseGrid.map(cell => Math.abs(cell.noiseValue.x)));
    const maxNoiseY = Math.max(1, ...this.noiseGrid.map(cell => Math.abs(cell.noiseValue.y)));
    this.noiseScaleX = priorBounds.width * 0.42 / maxNoiseX;
    this.noiseScaleY = priorBounds.height * 0.42 / maxNoiseY;
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
    const dataVariance = 0.18 ** 2;
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
      scaleX,
      scaleY
    };
  }

  buildDiffusionPaths() {
    const steps = parseInt(this.diffusionStepSelect.value, 10);
    const dt = 1 / steps;
    const center = { x: this.canvas.width * 0.12, y: this.canvas.height * 0.50 };
    this.diffusionSteps = steps;
    this.diffusionPaths = this.noiseGrid.map((cell, idx) => {
      const rng = mulberry32(this.seed * 100003 + idx * 7919 + 17);
      let pos = { ...cell.z0 };
      const path = [pos];

      for (let step = 0; step < steps; step++) {
        const tau = step / steps;
        const stats = this.getDiffusionStats(pos, tau);
        const driftX = 0.5 * stats.beta * stats.y.x + stats.beta * stats.score.x;
        const driftY = 0.5 * stats.beta * stats.y.y + stats.beta * stats.score.y;
        const [noiseX, noiseY] = normalPair(rng);
        const diffusionScale = Math.sqrt(stats.beta * dt);
        const nextY = {
          x: stats.y.x + driftX * dt + diffusionScale * noiseX,
          y: stats.y.y + driftY * dt + diffusionScale * noiseY
        };
        pos = {
          x: center.x + nextY.x * stats.scaleX,
          y: center.y + nextY.y * stats.scaleY
        };
        path.push(pos);
      }
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

  updateSelectedCellDetail() {
    const cell = this.noiseGrid[this.selectedCellIndex];
    if (!cell) return;

    const activeBreed = this.getActivePredictedBreed(this.currentTime);
    const velVector = this.getVelocityVector(cell, this.currentTime, this.selectedCellIndex);
    this.winningBreed = activeBreed;

    const isDiff = this.algorithm === 'diff';
    const pos = this.computeTrajectoryPos(cell.z0, cell.x1, this.currentTime, this.selectedCellIndex);
    const rgbZt = this.getCurrentRgbLatent(cell, pos);
    const confidence = isDiff
      ? Math.round(this.getDiffusionStats(pos, this.currentTime).responsibility * 100)
      : 100;
    const mainShortName = activeBreed.shortName || activeBreed.breedName;

    this.selectedCellName.textContent = `z₀-${cell.name} (시드 #${this.seed})`;
    this.selectedCellVal.textContent = `RGB z₀ = [${cell.rgbNoise.r >= 0 ? '+' : ''}${cell.rgbNoise.r.toFixed(2)}, ${cell.rgbNoise.g >= 0 ? '+' : ''}${cell.rgbNoise.g.toFixed(2)}, ${cell.rgbNoise.b >= 0 ? '+' : ''}${cell.rgbNoise.b.toFixed(2)}]`;
    this.selectedCellProjection.textContent = `2D 투영 π(z₀) = [${cell.noiseValue.x >= 0 ? '+' : ''}${cell.noiseValue.x.toFixed(2)}, ${cell.noiseValue.y >= 0 ? '+' : ''}${cell.noiseValue.y.toFixed(2)}]`;
    this.selectedCellTarget.textContent = isDiff ? `${mainShortName} (현재 score mode)` : `${cell.breedName} (조건부 pairing)`;

    if (this.metricRgbZt) this.metricRgbZt.textContent = `[${rgbZt.r >= 0 ? '+' : ''}${rgbZt.r.toFixed(2)}, ${rgbZt.g >= 0 ? '+' : ''}${rgbZt.g.toFixed(2)}, ${rgbZt.b >= 0 ? '+' : ''}${rgbZt.b.toFixed(2)}]`;
    if (this.metricVectorVal) this.metricVectorVal.textContent = `[x: ${velVector.vx > 0 ? '+' : ''}${velVector.vx.toFixed(1)}, y: ${velVector.vy > 0 ? '+' : ''}${velVector.vy.toFixed(1)}]`;
    if (this.metricVectorLabel) this.metricVectorLabel.innerHTML = isDiff ? 'reverse-SDE drift b<sub>τ</sub>:' : '조건부 목표 u<sub>τ</sub>:';
    if (this.metricVectorStatus) this.metricVectorStatus.textContent = isDiff ? 'score + VP drift (시간에 따라 변화)' : '선택한 한 쌍에서만 일정';
    if (this.metricNearestName) this.metricNearestName.textContent = `${mainShortName}`;
    if (this.metricConfidenceLabel) this.metricConfidenceLabel.textContent = isDiff ? 'mode posterior responsibility:' : '조건부 pairing:';
    if (this.metricConfidence) this.metricConfidence.textContent = `${confidence}%`;
    if (this.metricConfBar) this.metricConfBar.style.width = `${confidence}%`;
    if (this.targetBreedBadge) this.targetBreedBadge.textContent = isDiff ? `τ=${this.currentTime.toFixed(2)} score mode: ${mainShortName}` : `조건부 목적지: ${mainShortName}`;
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
      const breedData = this.algorithm === 'diff'
        ? this.getDiffusionStats(this.computeTrajectoryPos(cell.z0, cell.x1, t, this.selectedCellIndex), t).breed
        : cell;
      this.drawDenoisedCatOnContext(ectx, w, h, breedData, t);
    });
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
    const breedData = this.getActivePredictedBreed(this.currentTime) || this.noiseGrid[this.selectedCellIndex];
    if (!breedData) return;

    const ctx = this.targetCtx;
    const w = this.targetCanvas.width;
    const h = this.targetCanvas.height;

    ctx.clearRect(0, 0, w, h);

    this.drawDenoisedCatOnContext(ctx, w, h, breedData, this.currentTime);

    ctx.fillStyle = '#ffffff';
    ctx.font = '700 12px Outfit';
    ctx.textAlign = 'center';

    const mainShortName = breedData.shortName || breedData.breedName;
    const isDiff = this.algorithm === 'diff';
    const labelPrefix = isDiff ? `τ=${this.currentTime.toFixed(2)} score mode: ` : '조건부 목적지: ';
    
    ctx.fillText(`${labelPrefix}${mainShortName}`, w / 2, h - 10);

    this.renderEvolutionCanvases();
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
    const activeBreed = this.getActivePredictedBreed(this.currentTime);

    // 3. Breed Sub-Clusters on Manifold
    const activeClustersToDraw = this.algorithm === 'diff' ? this.allBreedFixedClusters : this.fixedBreedMap;

    if (activeClustersToDraw) {
      activeClustersToDraw.forEach((breedCluster) => {
        const isWinner = activeBreed && (activeBreed.breedName === breedCluster.name || activeBreed.name === breedCluster.name);
        
        ctx.save();
        ctx.strokeStyle = isWinner ? '#ec4899' : `${breedCluster.color}80`;
        ctx.fillStyle = isWinner ? 'rgba(236, 72, 153, 0.35)' : `${breedCluster.color}20`;
        ctx.lineWidth = isWinner ? 3.0 : 1.0;

        ctx.beginPath();
        ctx.arc(breedCluster.x1.x, breedCluster.x1.y, isWinner ? 18 : 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        if (isWinner) {
          ctx.shadowBlur = 15;
          ctx.shadowColor = '#ec4899';
          ctx.stroke();
        }

        ctx.fillStyle = isWinner ? '#ffffff' : '#9ca3af';
        ctx.font = isWinner ? '700 11px Outfit' : '500 9px Outfit';
        ctx.fillText(`${isWinner ? '🔥 ' : ''}${breedCluster.shortName}`, breedCluster.x1.x - 14, breedCluster.x1.y - 18);
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

    // 5. RENDER TRAJECTORIES & CLEAN SINGLE BREED NAME VECTOR BADGES (e.g. v_삼색이 [+42.1, -15.8])
    this.noiseGrid.forEach((cell, idx) => {
      const isSelectedCell = idx === this.selectedCellIndex;
      const pos = this.computeTrajectoryPos(cell.z0, cell.x1, this.currentTime, idx);
      const isFlow = this.algorithm === 'flow';
      const velVec = this.getVelocityVector(cell, this.currentTime, idx);

      ctx.save();
      ctx.lineWidth = isSelectedCell ? 3.0 : 1.2;

      if (isFlow) {
        ctx.strokeStyle = isSelectedCell ? '#00f0ff' : 'rgba(56, 189, 248, 0.35)';
        ctx.shadowBlur = isSelectedCell ? 10 : 0;
        ctx.shadowColor = '#00f0ff';
        
        ctx.beginPath();
        ctx.moveTo(cell.z0.x, cell.z0.y);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();

      } else {
        ctx.strokeStyle = isSelectedCell ? '#ff007f' : 'rgba(244, 63, 94, 0.35)';
        ctx.shadowBlur = isSelectedCell ? 10 : 0;
        ctx.shadowColor = '#ff007f';
        ctx.setLineDash([4, 4]);

        const path = this.diffusionPaths[idx];
        const completedStep = Math.floor(this.currentTime * this.diffusionSteps);
        ctx.beginPath();
        ctx.moveTo(path[0].x, path[0].y);
        for (let step = 1; step <= completedStep; step++) ctx.lineTo(path[step].x, path[step].y);
        ctx.stroke();
      }
      ctx.restore();

      // Initial Noise Dot z0
      ctx.save();
      ctx.fillStyle = `rgb(${cell.r}, ${cell.g}, ${cell.b})`;
      ctx.beginPath();
      ctx.arc(cell.z0.x, cell.z0.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Active Moving Dot x_t
      ctx.save();
      ctx.fillStyle = isFlow ? '#38bdf8' : '#f43f5e';
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isSelectedCell ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      if (this.currentTime < 0.98) {
        const arrowColor = isSelectedCell ? (isFlow ? '#00f0ff' : '#facc15') : (isFlow ? 'rgba(56, 189, 248, 0.75)' : 'rgba(244, 63, 94, 0.75)');
        const arrowLen = isSelectedCell ? 48 : 34;
        const strokeW = isSelectedCell ? 3.2 : 2.0;

        const vecNorm = Math.sqrt(velVec.vx * velVec.vx + velVec.vy * velVec.vy) || 1;
        const endX = pos.x + (velVec.vx / vecNorm) * arrowLen;
        const endY = pos.y + (velVec.vy / vecNorm) * arrowLen;

        this.drawArrowHead(ctx, pos.x, pos.y, endX, endY, arrowColor, strokeW);

        if (isSelectedCell) {
          const realtimeTarget = this.getRealtimeParticleTarget(cell, this.currentTime, idx);
          const singleBreedName = isFlow ? cell.name : (realtimeTarget.shortName || realtimeTarget.name);
          const vecName = isFlow ? `u_${singleBreedName}` : `b_${singleBreedName}`;

          ctx.save();
          ctx.fillStyle = arrowColor;
          ctx.font = '700 11px JetBrains Mono';
          ctx.shadowBlur = 8;
          ctx.shadowColor = arrowColor;
          ctx.fillText(`${vecName} [${velVectorStr(velVec.vx)}, ${velVectorStr(velVec.vy)}]`, endX + 4, endY - 6);
          ctx.restore();
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

  computeTrajectoryPos(z0, x1, t, seedOffset) {
    if (this.algorithm === 'flow') {
      return {
        x: (1 - t) * z0.x + t * x1.x,
        y: (1 - t) * z0.y + t * x1.y
      };
    }

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

  selfCheck() {
    const path = this.diffusionPaths[0];
    const stats = this.getDiffusionStats(path[Math.floor(path.length / 2)], 0.5);
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
    if (
      path.length !== this.diffusionSteps + 1 ||
      !Number.isFinite(stats.score.x) ||
      !Number.isFinite(stats.score.y) ||
      stats.responsibility <= 0 ||
      stats.responsibility > 1 ||
      Math.abs(cell.noiseValue.x - projectedX) > 1e-12 ||
      Math.abs(cell.noiseValue.y - projectedY) > 1e-12 ||
      Math.abs(cell.rgbNoise.r - rgbAtStart.r) > 1e-12 ||
      Math.abs(cell.rgbNoise.g - rgbAtStart.g) > 1e-12 ||
      Math.abs(cell.rgbNoise.b - rgbAtStart.b) > 1e-12 ||
      !allNoiseInsidePrior
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
