/**
 * Generative Model Explainer Engine
 * Dynamic SDE Multi-Breed Manifold Sampling!
 * In Diffusion SDE mode, SDE stochastic drift dynamically samples fresh diverse target breeds
 * from the full 20-species pool across diverse manifold coordinates as time advances!
 */

function mulberry32(a) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
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
    this.selectedCellTarget = document.getElementById('selected-cell-target');
    this.targetBreedBadge = document.getElementById('target-breed-badge');
    this.predictedX0Status = document.getElementById('predicted-x0-status');
    
    // Velocity Vector Metric DOM
    this.metricVectorVal = document.getElementById('metric-vector-val');
    this.metricVectorStatus = document.getElementById('metric-vector-status');
    this.metricNearestName = document.getElementById('metric-nearest-name');
    this.metricConfidence = document.getElementById('metric-confidence');
    this.metricConfBar = document.getElementById('metric-conf-bar');

    this.btnModeFlow = document.getElementById('btn-mode-flow');
    this.btnModeDiff = document.getElementById('btn-mode-diff');

    this.btnPlayPause = document.getElementById('btn-play-pause');
    this.btnReset = document.getElementById('btn-reset');
    this.scrubber = document.getElementById('scrubber');
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
    this.fixedFlowWinner = null;
    
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
      this.renderManifold();
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
      this.predictedX0Status.textContent = "플로우 매칭 모드: t=0 노이즈에서 시작해 고유 위치의 고양이 특징이 일직선으로 선명해집니다.";
      this.updateSelectedCellDetail();
      this.renderTargetImage();
    });

    this.btnModeDiff.addEventListener('click', () => {
      this.algorithm = 'diff';
      this.btnModeDiff.classList.add('active');
      this.btnModeFlow.classList.remove('active');
      this.predictedX0Status.textContent = "디퓨전 SDE 모드: 확률적 뜀박질에 따라 20개 모든 고양이 품종이 다양한 위치에서 동적으로 탐색됩니다!";
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
    const rng = mulberry32(this.seed * 9999 + (this.prompt === 'cat' ? 100 : this.prompt === 'dog' ? 200 : 300));
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
      ],
      car: [
        { id: 0, name: '레드 슈퍼카', shortName: '레드슈퍼카', color: '#f43f5e', pattern: 'super' },
        { id: 1, name: '사이버 트럭', shortName: '사이버트럭', color: '#94a3b8', pattern: 'truck' },
        { id: 2, name: '클래식 세단', shortName: '클래식세단', color: '#3b82f6', pattern: 'sedan' },
        { id: 3, name: '오프로드 SUV', shortName: '오프로드SUV', color: '#84cc16', pattern: 'suv' },
        { id: 4, name: '미래형 컨셉카', shortName: '컨셉카', color: '#a855f7', pattern: 'concept' },
        { id: 5, name: 'F1 레이싱카', shortName: 'F1레이싱카', color: '#eab308', pattern: 'f1' },
        { id: 6, name: '옐로우 컨버터블', shortName: '컨버터블', color: '#ec4899', pattern: 'open' },
        { id: 7, name: '빈티지 로드스터', shortName: '로드스터', color: '#f97316', pattern: 'vintage' },
        { id: 8, name: '전기 럭셔리 픽업', shortName: '전기픽업', color: '#0ea5e9', pattern: 'truck' },
        { id: 9, name: '에어로 하이퍼카', shortName: '하이퍼카', color: '#10b981', pattern: 'super' },
        { id: 10, name: '랠리 크로스카', shortName: '랠리카', color: '#facc15', pattern: 'f1' },
        { id: 11, name: '몬스터 트럭', shortName: '몬스터트럭', color: '#ef4444', pattern: 'suv' },
        { id: 12, name: '미니 쿠퍼', shortName: '미니쿠퍼', color: '#06b6d4', pattern: 'sedan' },
        { id: 13, name: '경찰 파트롤 세단', shortName: '경찰세단', color: '#38bdf8', pattern: 'sedan' },
        { id: 14, name: '소방 구급 트럭', shortName: '소방트럭', color: '#dc2626', pattern: 'truck' },
        { id: 15, name: '옐로우 택시', shortName: '옐로우택시', color: '#eab308', pattern: 'sedan' },
        { id: 16, name: '스쿨 버스', shortName: '스쿨버스', color: '#f59e0b', pattern: 'suv' },
        { id: 17, name: '캠핑 버스', shortName: '캠핑버스', color: '#84cc16', pattern: 'suv' },
        { id: 18, name: '자율주행 로보택시', shortName: '로보택시', color: '#a855f7', pattern: 'concept' },
        { id: 19, name: '스노우 모빌', shortName: '스노우모빌', color: '#e2e8f0', pattern: 'concept' }
      ]
    };

    const fullPool = richPools[this.prompt];
    const w = this.canvas.width;
    const h = this.canvas.height;

    const promptCenters = {
      cat: { x: w * 0.65, y: h * 0.45, radius: 130 },
      dog: { x: w * 0.62, y: h * 0.75, radius: 120 },
      car: { x: w * 0.35, y: h * 0.70, radius: 125 }
    };
    const classCenter = promptCenters[this.prompt];

    // EVERY SINGLE BREED IN THE 20-SPECIES POOL HAS ITS OWN UNIQUE DEDICATED (x, y) POSITION ON THE MANIFOLD!
    const allBreedFixedClusters = fullPool.map((breed) => {
      const angle = (breed.id / 20) * Math.PI * 2;
      const radiusDist = classCenter.radius * (0.55 + ((breed.id % 3) * 0.15));
      return {
        ...breed,
        x1: {
          x: classCenter.x + Math.cos(angle) * radiusDist,
          y: classCenter.y + Math.sin(angle) * radiusDist
        }
      };
    });

    const sampledPool = [...allBreedFixedClusters].sort(() => rng() - 0.5).slice(0, 9);
    const winnerIndex = Math.floor(rng() * 9);

    for (let i = 0; i < 9; i++) {
      const r = Math.floor(rng() * 210 + 45);
      const g = Math.floor(rng() * 210 + 45);
      const b = Math.floor(rng() * 210 + 45);

      const row = Math.floor(i / 3);
      const col = i % 3;
      const z0x = w * 0.12 + (col - 1) * 35 + (rng() - 0.5) * 15;
      const z0y = h * 0.50 + (row - 1) * 35 + (rng() - 0.5) * 15;

      const targetBreed = sampledPool[i];

      this.noiseGrid.push({
        name: targetBreed.shortName,
        fullName: targetBreed.name,
        r, g, b,
        z0: { x: z0x, y: z0y },
        x1: targetBreed.x1,
        breedName: targetBreed.name,
        targetColor: targetBreed.color,
        pattern: targetBreed.pattern,
        isSeedWinner: i === winnerIndex
      });
    }

    this.fixedBreedMap = sampledPool;
    this.allBreedFixedClusters = allBreedFixedClusters;
    this.fullPool = fullPool;
    this.fixedFlowWinner = this.noiseGrid[winnerIndex];

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
        <span class="cell-name">${cell.name}</span>
        <span class="cell-val">[${cell.r}, ${cell.g}, ${cell.b}]</span>
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

  getTwoNearbyNeighborNames(targetBreedData) {
    if (!targetBreedData || !this.allBreedFixedClusters) return '';

    const targetPos = targetBreedData.x1;
    const sortedNeighbors = [...this.allBreedFixedClusters]
      .filter(b => b.shortName !== targetBreedData.shortName && b.name !== targetBreedData.name)
      .map(b => {
        const dx = b.x1.x - targetPos.x;
        const dy = b.x1.y - targetPos.y;
        return { shortName: b.shortName, dist: Math.sqrt(dx * dx + dy * dy) };
      })
      .sort((a, b) => a.dist - b.dist);

    const neighbor1 = sortedNeighbors[0] ? sortedNeighbors[0].shortName : '';
    const neighbor2 = sortedNeighbors[1] ? sortedNeighbors[1].shortName : '';

    if (neighbor1 && neighbor2) return ` (+ ${neighbor1}, + ${neighbor2})`;
    if (neighbor1) return ` (+ ${neighbor1})`;
    return '';
  }

  // DYNAMIC SDE STEP TARGET PREDICTION:
  // In Diffusion SDE mode, SDE Brownian drift wanders across ALL 20 BREEDS in the pool at different step phases!
  getActivePredictedBreed(t) {
    if (this.algorithm === 'flow' || t >= 0.85) {
      return this.fixedFlowWinner;
    } else if (t < 0.05) {
      return this.noiseGrid[this.selectedCellIndex] || this.fixedFlowWinner;
    } else {
      // In Diffusion SDE mode, SDE drift wanders across the FULL 20-BREED POOL at different SDE step phases!
      const stepQuantum = Math.floor(t * 14);
      const stepRng = mulberry32(this.seed * 777 + stepQuantum * 1337);
      const randomFullIdx = Math.floor(stepRng() * this.allBreedFixedClusters.length);
      return this.allBreedFixedClusters[randomFullIdx] || this.fixedFlowWinner;
    }
  }

  getRealtimeParticleTarget(cell, t, idx) {
    if (this.algorithm === 'flow' || t < 0.05 || t >= 0.85) {
      return cell;
    } else {
      const stepQuantum = Math.floor(t * 14 + idx * 3);
      const stepRng = mulberry32(this.seed * 777 + stepQuantum * 1337 + idx * 999);
      const randFullIdx = Math.floor(stepRng() * this.allBreedFixedClusters.length);
      return this.allBreedFixedClusters[randFullIdx] || cell;
    }
  }

  getCombinedCenterPoint(t) {
    let sumX = 0;
    let sumY = 0;
    let totalWeight = 0;

    const activeBreed = this.getActivePredictedBreed(t);

    for (let i = 0; i < this.noiseGrid.length; i++) {
      const cell = this.noiseGrid[i];
      const pos = this.computeTrajectoryPos(cell.z0, cell.x1, t, i);

      const isWinner = cell.breedName === activeBreed.breedName;
      const weight = isWinner ? (1.0 + t * 6.0) : 1.0;

      sumX += pos.x * weight;
      sumY += pos.y * weight;
      totalWeight += weight;
    }

    return { x: sumX / totalWeight, y: sumY / totalWeight };
  }

  getVelocityVector(cell, t, idx) {
    const pos = this.computeTrajectoryPos(cell.z0, cell.x1, t, idx);

    if (this.algorithm === 'flow') {
      return {
        vx: (cell.x1.x - cell.z0.x) * 0.15,
        vy: (cell.x1.y - cell.z0.y) * 0.15,
        isConstant: true
      };
    } else {
      const targetPt = this.getRealtimeParticleTarget(cell, t, idx).x1;

      const dirX = targetPt.x - pos.x;
      const dirY = targetPt.y - pos.y;
      const dist = Math.sqrt(dirX * dirX + dirY * dirY) || 1;

      const speedMagnitude = Math.min(65, dist * 0.45);
      return {
        vx: (dirX / dist) * speedMagnitude,
        vy: (dirY / dist) * speedMagnitude,
        isConstant: false
      };
    }
  }

  updateSelectedCellDetail() {
    const cell = this.noiseGrid[this.selectedCellIndex];
    if (!cell) return;

    const activeBreed = this.getActivePredictedBreed(this.currentTime);
    const centerPt = this.getCombinedCenterPoint(this.currentTime);
    const velVector = this.getVelocityVector(cell, this.currentTime, this.selectedCellIndex);

    const minDist = Math.sqrt((centerPt.x - activeBreed.x1.x)**2 + (centerPt.y - activeBreed.x1.y)**2);
    this.winningBreed = activeBreed;

    const confidence = Math.max(0, Math.min(100, Math.floor(100 - (minDist / 1.8))));
    const mainShortName = activeBreed.shortName || activeBreed.breedName;
    const neighborText = this.getTwoNearbyNeighborNames(activeBreed);

    this.selectedCellName.textContent = `${cell.name} 임베딩 (시드 #${this.seed})`;
    this.selectedCellVal.textContent = `초기 노이즈 z0 [${cell.r}, ${cell.g}, ${cell.b}]`;
    this.selectedCellTarget.textContent = `${cell.breedName}`;

    const isDiff = this.algorithm === 'diff';
    const isEarly = this.currentTime < 0.85;
    const labelPrefix = (isDiff && isEarly) ? `[t=${this.currentTime.toFixed(2)} 예측] ` : '';

    if (this.metricVectorVal) this.metricVectorVal.textContent = `[v_x: ${velVector.vx > 0 ? '+' : ''}${velVector.vx.toFixed(1)}, v_y: ${velVector.vy > 0 ? '+' : ''}${velVector.vy.toFixed(1)}]`;
    if (this.metricVectorStatus) this.metricVectorStatus.textContent = velVector.isConstant ? `${cell.name} 고정 속도장 (Constant OT Flow)` : `${cell.name} 실시간 회전 속도장 (Dynamic SDE Drift)`;
    if (this.metricNearestName) this.metricNearestName.textContent = `${labelPrefix}${mainShortName}`;
    if (this.metricConfidence) this.metricConfidence.textContent = `${confidence}%`;
    if (this.metricConfBar) this.metricConfBar.style.width = `${confidence}%`;
    if (this.targetBreedBadge) this.targetBreedBadge.textContent = `${labelPrefix}최고 밀도: ${mainShortName}${neighborText}`;
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

      const cell = this.noiseGrid[this.selectedCellIndex] || this.fixedFlowWinner;
      if (!cell) return;

      this.drawDenoisedCatOnContext(ectx, w, h, cell, t);
    });
  }

  drawDenoisedCatOnContext(ctx, w, h, breedData, t) {
    const isNoisePhase = t < 0.05;

    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, w, h);

    if (isNoisePhase) {
      const imgData = ctx.createImageData(w, h);
      const data = imgData.data;
      for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.floor(Math.random() * 255);
        data[i+1] = Math.floor(Math.random() * 255);
        data[i+2] = Math.floor(Math.random() * 255);
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

    } else {
      ctx.fillStyle = mainColor;
      if (t >= 0.20) {
        ctx.fillRect(-75, 5, 150, 28);
        ctx.beginPath(); ctx.arc(-5, 5, 38, Math.PI, 0); ctx.fill();
      }
      if (t >= 0.55) {
        ctx.fillStyle = '#334155';
        ctx.beginPath();
        ctx.arc(-45, 30, 14, 0, Math.PI * 2);
        ctx.arc(45, 30, 14, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();

    if (t < 0.95) {
      const noiseAlpha = (1 - t) * 0.75;
      const overlayData = ctx.getImageData(0, 0, w, h);
      const pix = overlayData.data;
      for (let i = 0; i < pix.length; i += 4) {
        if (Math.random() < noiseAlpha) {
          const grain = (Math.random() - 0.5) * 180;
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
    const neighborText = this.getTwoNearbyNeighborNames(breedData);

    const isDiff = this.algorithm === 'diff';
    const isEarly = this.currentTime < 0.85;
    const labelPrefix = (isDiff && isEarly) ? `[t=${this.currentTime.toFixed(2)} 예측] ` : '★ 최종 생성: ';
    
    ctx.fillText(`${labelPrefix}${mainShortName}${neighborText}`, w / 2, h - 10);

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
    ctx.ellipse(w * 0.5, h * 0.55, w * 0.42, h * 0.38, -0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#a5b4fc';
    ctx.font = '600 13px Outfit';
    ctx.fillText('전체 모델 데이터 분포 p(x) [개, 고양이, 자동차...]', w * 0.15, h * 0.20);
    ctx.restore();

    // 2. Prompt Class Distributions
    const promptConfigs = [
      { id: 'cat', label: '고양이 분포 p(x|"고양이")', center: { x: w * 0.65, y: h * 0.45 }, rx: 130, ry: 95, color: '#38bdf8' },
      { id: 'dog', label: '강아지 분포 p(x|"강아지")', center: { x: w * 0.62, y: h * 0.75 }, rx: 120, ry: 80, color: '#f59e0b' },
      { id: 'car', label: '자동차 분포 p(x|"자동차")', center: { x: w * 0.35, y: h * 0.70 }, rx: 110, ry: 75, color: '#ec4899' }
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
        ctx.fillText(`★ ${cfg.label}`, cfg.center.x - 70, cfg.center.y - cfg.ry + 18);
      }
      ctx.restore();
    });

    // Calculate Dynamic Combined Batch Centroid (\bar{x}_{center})
    const centerPt = this.getCombinedCenterPoint(this.currentTime);
    const activeBreed = this.getActivePredictedBreed(this.currentTime);

    // 3. Breed Sub-Clusters on Manifold
    // In Diffusion SDE mode, renders active target breed clusters dynamically across ALL 20 SPECIES LOCATIONS!
    const activeClustersToDraw = (this.algorithm === 'diff' && this.currentTime < 0.85) ? this.allBreedFixedClusters : this.fixedBreedMap;

    if (activeClustersToDraw) {
      activeClustersToDraw.forEach((breedCluster) => {
        const isWinner = activeBreed && (activeBreed.name === breedCluster.name || activeBreed.shortName === breedCluster.shortName);
        
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
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.rect(w * 0.05, h * 0.35, 150, 150);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 12px JetBrains Mono';
    ctx.fillText(`초기 노이즈 z0 (시드 #${this.seed})`, w * 0.06, h * 0.33);
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

        ctx.beginPath();
        for (let tStep = 0; tStep <= this.currentTime; tStep += 0.04) {
          const stepPos = this.computeTrajectoryPos(cell.z0, cell.x1, tStep, idx);
          if (tStep === 0) ctx.moveTo(stepPos.x, stepPos.y);
          else ctx.lineTo(stepPos.x, stepPos.y);
        }
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

        const realtimeTarget = this.getRealtimeParticleTarget(cell, this.currentTime, idx);
        const singleBreedName = isFlow ? cell.name : (realtimeTarget.shortName || realtimeTarget.name);

        ctx.save();
        ctx.fillStyle = arrowColor;
        ctx.font = isSelectedCell ? '700 11px JetBrains Mono' : '500 9px JetBrains Mono';
        ctx.shadowBlur = isSelectedCell ? 8 : 0;
        ctx.shadowColor = arrowColor;
        
        const isDiffEarly = (!isFlow) && (this.currentTime >= 0.05) && (this.currentTime < 0.85);
        const vecNumText = `v_${singleBreedName}${isDiffEarly ? '(예측)' : ''} [${velVec.vx > 0 ? '+' : ''}${velVec.vx.toFixed(1)}, ${velVec.vy > 0 ? '+' : ''}${velVectorStr(velVec.vy)}]`;
        
        ctx.fillText(vecNumText, endX + 4, endY + (idx % 2 === 0 ? -6 : 10));
        ctx.restore();
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
      ctx.fillText('★ 통합 중심점 (x̄)', centerPt.x - 40, centerPt.y - 14);

      ctx.restore();
    }
  }

  computeTrajectoryPos(z0, x1, t, seedOffset) {
    if (this.algorithm === 'flow') {
      return {
        x: (1 - t) * z0.x + t * x1.x,
        y: (1 - t) * z0.y + t * x1.y
      };
    } else {
      const meanX = (1 - Math.pow(t, 0.65)) * z0.x + Math.pow(t, 0.65) * x1.x;
      const meanY = (1 - Math.pow(t, 0.65)) * z0.y + Math.pow(t, 0.65) * x1.y;

      const noiseLevel = (1 - t) * 16.0;
      const stepPhase = Math.floor(t * 12 + seedOffset * 1.7);
      
      const rng1 = mulberry32(this.seed * 333 + stepPhase * 999 + seedOffset * 42);
      const rng2 = mulberry32(this.seed * 444 + stepPhase * 888 + seedOffset * 17);

      const smoothJitterX = (rng1() - 0.5) * 2.0 * noiseLevel;
      const smoothJitterY = (rng2() - 0.5) * 2.0 * noiseLevel;

      return {
        x: meanX + smoothJitterX,
        y: meanY + smoothJitterY
      };
    }
  }

  hexToRgb(hex) {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : { r: 255, g: 255, b: 255 };
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
