<template>
  <div dir="ltr" class="retro-cockpit">
    <!-- Scanlines overlay for retro effect -->
    <div class="scanlines"></div>

    <!-- Title -->
    <div class="title">
      <span class="title-text">GPU COMMAND CENTER</span>
      <span class="title-blink">●</span>
    </div>

    <!-- Main dashboard -->
    <div class="dashboard">
      <!-- Left gauges -->
      <div class="gauge-column left">
        <!-- Power gauge (speedometer style) -->
        <div class="gauge power-gauge">
          <div class="gauge-label">POWER</div>
          <div class="gauge-ring">
            <svg viewBox="0 0 100 100">
              <circle class="gauge-bg" cx="50" cy="50" r="40" />
              <circle
                class="gauge-fill power-fill"
                cx="50" cy="50" r="40"
                :style="{ strokeDashoffset: powerOffset }"
              />
            </svg>
            <div class="gauge-value">{{ Math.round(powerWatts) }}</div>
            <div class="gauge-unit">WATTS</div>
          </div>
          <div class="gauge-ticks">
            <span v-for="n in 5" :key="n" class="tick">{{ (n - 1) * (powerLimit / 4) }}</span>
          </div>
        </div>

        <!-- Temperature gauge -->
        <div class="gauge temp-gauge">
          <div class="gauge-label">TEMP</div>
          <div class="thermometer">
            <div class="thermo-tube">
              <div class="thermo-fill" :style="{ height: `${tempPercent}%` }" :class="{ hot: temperature > 70, critical: temperature > 80 }"></div>
              <div class="thermo-marks">
                <span v-for="n in 5" :key="n">{{ 100 - (n - 1) * 20 }}°</span>
              </div>
            </div>
            <div class="thermo-bulb" :class="{ hot: temperature > 70 }"></div>
          </div>
          <div class="gauge-value-small">{{ temperature }}°C</div>
        </div>
      </div>

      <!-- Center display -->
      <div class="center-display">
        <!-- GPU Name -->
        <div class="gpu-name">{{ gpuName }}</div>

        <!-- Main status -->
        <div class="status-display">
          <div class="status-row">
            <span class="status-label">UTILIZATION</span>
            <div class="status-bar">
              <div class="status-bar-fill" :style="{ width: `${utilization}%` }"></div>
            </div>
            <span class="status-value">{{ utilization }}%</span>
          </div>
          <div class="status-row">
            <span class="status-label">PROGRESS</span>
            <div class="status-bar progress-bar">
              <div class="status-bar-fill progress-fill" :style="{ width: `${progress}%` }"></div>
            </div>
            <span class="status-value">{{ Math.round(progress) }}%</span>
          </div>
        </div>

        <!-- Road animation -->
        <div class="road-container">
          <div class="road" :class="{ moving: progress > 0 && progress < 100 }">
            <div class="road-line" v-for="n in 8" :key="n"></div>
          </div>
          <div class="horizon"></div>
        </div>
      </div>

      <!-- Right gauges -->
      <div class="gauge-column right">
        <!-- VRAM gauge (fuel style) -->
        <div class="gauge vram-gauge">
          <div class="gauge-label">VRAM</div>
          <div class="fuel-tank">
            <div class="fuel-level" :style="{ height: `${vramPercent}%` }"></div>
            <div class="fuel-marks">
              <span>F</span>
              <span>¾</span>
              <span>½</span>
              <span>¼</span>
              <span>E</span>
            </div>
          </div>
          <div class="gauge-value-small">{{ usedGb }}/{{ totalGb }}GB</div>
        </div>

        <!-- CO2 counter -->
        <div class="gauge co2-gauge">
          <div class="gauge-label">EMISSIONS</div>
          <div class="digital-display">
            <span class="digit">{{ co2Display }}</span>
          </div>
          <div class="gauge-unit">g CO₂</div>
        </div>
      </div>
    </div>

    <!-- Bottom info bar -->
    <div class="info-bar">
      <span class="info-item">
        <span class="info-label">ENERGY:</span>
        <span class="info-value">{{ energyDisplay }} kWh</span>
      </span>
      <span class="info-item">
        <span class="info-label">RUNTIME:</span>
        <span class="info-value">{{ formatTime(elapsedSeconds) }}</span>
      </span>
      <span class="info-item">
        <span class="info-label">TDP:</span>
        <span class="info-value">{{ powerLimit }}W</span>
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  progress: number
  powerWatts: number
  powerLimit: number
  temperature: number
  utilization: number
  vramUsedMb: number
  vramTotalMb: number
  co2Grams: number
  energyWh: number
  elapsedSeconds: number
  gpuName: string
}>()

// Gauge calculations
const powerOffset = computed(() => {
  const circumference = 2 * Math.PI * 40
  const ratio = Math.min(1, props.powerWatts / props.powerLimit)
  return circumference * (1 - ratio * 0.75) // 3/4 circle
})

const tempPercent = computed(() => Math.min(100, Math.max(0, props.temperature)))

const vramPercent = computed(() => {
  if (props.vramTotalMb === 0) return 0
  return (props.vramUsedMb / props.vramTotalMb) * 100
})

const usedGb = computed(() => Math.round(props.vramUsedMb / 1024))
const totalGb = computed(() => Math.round(props.vramTotalMb / 1024))

const co2Display = computed(() => props.co2Grams.toFixed(2).padStart(7, '0'))
const energyDisplay = computed(() => (props.energyWh / 1000).toFixed(4))

const temperature = computed(() => props.temperature || 0)
const utilization = computed(() => props.utilization || 0)

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}
</script>

<style scoped>
.retro-cockpit {
  width: 100%;
  height: 320px;
  background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
  border-radius: 12px;
  position: relative;
  overflow: hidden;
  font-family: 'Courier New', monospace;
  color: #0f0;
}

/* Scanlines */
.scanlines {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: repeating-linear-gradient(
    0deg,
    rgba(0, 0, 0, 0.1) 0px,
    rgba(0, 0, 0, 0.1) 1px,
    transparent 1px,
    transparent 2px
  );
  pointer-events: none;
  z-index: 100;
}

/* Title */
.title {
  text-align: center;
  padding: 8px;
  border-bottom: 1px solid #0f03;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
}

.title-text {
  font-size: 14px;
  letter-spacing: 3px;
  text-shadow: 0 0 10px #0f0;
}

.title-blink {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* Dashboard */
.dashboard {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  height: calc(100% - 80px);
}

.gauge-column {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100px;
}

/* Gauges */
.gauge {
  background: rgba(0, 50, 0, 0.3);
  border: 1px solid #0f04;
  border-radius: 8px;
  padding: 8px;
  text-align: center;
}

.gauge-label {
  font-size: 10px;
  color: #0f0;
  margin-bottom: 5px;
  letter-spacing: 1px;
}

/* Power gauge (circular) */
.gauge-ring {
  position: relative;
  width: 70px;
  height: 70px;
  margin: 0 auto;
}

.gauge-ring svg {
  transform: rotate(-135deg);
}

.gauge-bg {
  fill: none;
  stroke: #0f02;
  stroke-width: 8;
}

.gauge-fill {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
  stroke-dasharray: 188.5; /* 2 * PI * 40 * 0.75 */
  transition: stroke-dashoffset 0.5s ease;
}

.power-fill {
  stroke: #0f0;
  filter: drop-shadow(0 0 5px #0f0);
}

.gauge-value {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 16px;
  font-weight: bold;
}

.gauge-unit {
  font-size: 8px;
  color: #0f08;
  margin-top: 2px;
}

.gauge-value-small {
  font-size: 11px;
  margin-top: 5px;
}

/* Thermometer */
.thermometer {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  height: 60px;
  gap: 5px;
}

.thermo-tube {
  width: 12px;
  height: 50px;
  background: #0f02;
  border-radius: 6px;
  position: relative;
  overflow: hidden;
}

.thermo-fill {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: #0f0;
  border-radius: 0 0 6px 6px;
  transition: height 0.5s ease, background 0.3s ease;
}

.thermo-fill.hot {
  background: #ff0;
}

.thermo-fill.critical {
  background: #f00;
  animation: pulse-critical 0.5s infinite;
}

@keyframes pulse-critical {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.thermo-bulb {
  width: 16px;
  height: 16px;
  background: #0f0;
  border-radius: 50%;
  transition: background 0.3s ease;
}

.thermo-bulb.hot {
  background: #f00;
  box-shadow: 0 0 10px #f00;
}

.thermo-marks {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: 50px;
  font-size: 7px;
  color: #0f06;
}

/* Center display */
.center-display {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0 15px;
}

.gpu-name {
  font-size: 10px;
  color: #0f08;
  margin-bottom: 8px;
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status-display {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-label {
  font-size: 9px;
  width: 70px;
  color: #0f08;
}

.status-bar {
  flex: 1;
  height: 10px;
  background: #0f02;
  border-radius: 5px;
  overflow: hidden;
}

.status-bar-fill {
  height: 100%;
  background: #0f0;
  border-radius: 5px;
  transition: width 0.3s ease;
  box-shadow: 0 0 5px #0f0;
}

.progress-fill {
  background: linear-gradient(90deg, #0f0 0%, #ff0 50%, #f90 100%);
}

.status-value {
  font-size: 11px;
  width: 35px;
  text-align: right;
}

/* Road animation */
.road-container {
  flex: 1;
  width: 100%;
  position: relative;
  margin-top: 10px;
  perspective: 200px;
  overflow: hidden;
}

.road {
  position: absolute;
  bottom: 0;
  left: 10%;
  right: 10%;
  height: 80px;
  background: #222;
  transform: rotateX(60deg);
  transform-origin: bottom;
}

.road-line {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  width: 4px;
  height: 15px;
  background: #ff0;
  opacity: 0;
}

.road.moving .road-line {
  animation: road-move 1s linear infinite;
}

.road-line:nth-child(1) { top: 10%; animation-delay: 0s; }
.road-line:nth-child(2) { top: 20%; animation-delay: 0.125s; }
.road-line:nth-child(3) { top: 30%; animation-delay: 0.25s; }
.road-line:nth-child(4) { top: 40%; animation-delay: 0.375s; }
.road-line:nth-child(5) { top: 50%; animation-delay: 0.5s; }
.road-line:nth-child(6) { top: 60%; animation-delay: 0.625s; }
.road-line:nth-child(7) { top: 70%; animation-delay: 0.75s; }
.road-line:nth-child(8) { top: 80%; animation-delay: 0.875s; }

@keyframes road-move {
  0% { opacity: 0; transform: translateX(-50%) translateY(-50px) scaleY(0.5); }
  20% { opacity: 1; }
  80% { opacity: 1; }
  100% { opacity: 0; transform: translateX(-50%) translateY(30px) scaleY(2); }
}

.horizon {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, #0f06, transparent);
}

/* VRAM fuel tank */
.fuel-tank {
  width: 30px;
  height: 60px;
  background: #0f02;
  border: 1px solid #0f04;
  border-radius: 4px;
  margin: 0 auto;
  position: relative;
  overflow: hidden;
}

.fuel-level {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(180deg, #0ff 0%, #00f 100%);
  transition: height 0.5s ease;
  box-shadow: 0 0 10px #0ff;
}

.fuel-marks {
  position: absolute;
  top: 0;
  right: -20px;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  font-size: 7px;
  color: #0f06;
}

/* CO2 digital display */
.digital-display {
  background: #000;
  border: 1px solid #0f04;
  padding: 5px 8px;
  border-radius: 4px;
  margin: 5px 0;
}

.digit {
  font-family: 'Courier New', monospace;
  font-size: 14px;
  color: #f00;
  text-shadow: 0 0 5px #f00;
  letter-spacing: 1px;
}

/* Info bar */
.info-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-around;
  padding: 8px;
  background: rgba(0, 50, 0, 0.3);
  border-top: 1px solid #0f03;
}

.info-item {
  display: flex;
  gap: 5px;
  font-size: 10px;
}

.info-label {
  color: #0f06;
}

.info-value {
  color: #0f0;
}
</style>
