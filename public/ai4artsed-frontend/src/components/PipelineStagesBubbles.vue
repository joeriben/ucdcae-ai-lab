<template>
  <div dir="ltr" class="pipeline-stages">
    <!-- Stage Bubbles -->
    <div
      v-for="(stage, index) in stages"
      :key="stage.name"
      class="stage-item"
      :style="getStageStyle(index)"
    >
      <!-- Stage Bubble -->
      <div
        class="stage-bubble"
        :class="[`status-${stage.status}`]"
        :style="{ background: getStageColor(stage) }"
      >
        <div class="bubble-content">
          <!-- Stage Icon/Label -->
          <div class="stage-label">{{ stage.label }}</div>

          <!-- Preview Text -->
          <div v-if="stage.previewText" class="stage-preview">
            {{ stage.previewText }}
          </div>

          <!-- Processing Indicator -->
          <div v-if="stage.status === 'processing'" class="processing-spinner"></div>

          <!-- Completed Check -->
          <div v-if="stage.status === 'completed'" class="completed-check">✓</div>
        </div>
      </div>

      <!-- Connection Arrow -->
      <div v-if="index < stages.length - 1" class="stage-arrow">→</div>
    </div>

    <!-- Final Output Image (if available) -->
    <div v-if="outputImage" class="output-image-container" :style="getOutputImageStyle()">
      <img
        :src="outputImage"
        alt="Generated Output"
        class="output-image"
        @click="$emit('imageClick', outputImage)"
      />
      <div class="image-label">OUTPUT</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

type StageStatus = 'waiting' | 'processing' | 'completed'

interface Stage {
  name: string
  label: string
  status: StageStatus
  previewText?: string
  color: string
}

interface Props {
  stages: Stage[]
  outputImage?: string | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  imageClick: [imageUrl: string]
}>()

// Calculate horizontal positioning for each stage
function getStageStyle(index: number) {
  const totalStages = props.stages.length
  const spacing = 80 / (totalStages - 1 || 1) // Distribute across 80% of width
  const leftPosition = 10 + (spacing * index) // Start at 10%, end at 90%

  return {
    left: `${leftPosition}%`,
    top: '50%'
  }
}

// Get stage color based on status
function getStageColor(stage: Stage): string {
  if (stage.status === 'waiting') {
    return '#CCCCCC'
  }
  return stage.color
}

// Position for output image
function getOutputImageStyle() {
  return {
    right: '5%',
    top: '50%'
  }
}
</script>

<style scoped>
.pipeline-stages {
  position: relative;
  width: 100%;
  height: 100%;
}

.stage-item {
  position: absolute;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  gap: 1rem;
}

.stage-bubble {
  width: 12vw;
  height: 12vw;
  max-width: 150px;
  max-height: 150px;
  min-width: 100px;
  min-height: 100px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  transition: all 0.5s ease;
  position: relative;
}

.bubble-content {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  text-align: center;
  color: white;
  position: relative;
}

.stage-label {
  font-weight: 700;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stage-preview {
  font-size: 0.7rem;
  line-height: 1.2;
  opacity: 0.95;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

/* Status: waiting */
.status-waiting .bubble-content {
  color: #666;
}

/* Status: processing */
.status-processing {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

.processing-spinner {
  position: absolute;
  width: 2rem;
  height: 2rem;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* Status: completed */
.completed-check {
  position: absolute;
  top: 10%;
  right: 10%;
  width: 1.5rem;
  height: 1.5rem;
  background: rgba(255, 255, 255, 0.9);
  color: #4CAF50;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: bold;
  animation: check-pop 0.3s ease;
}

@keyframes check-pop {
  0% {
    transform: scale(0);
  }
  50% {
    transform: scale(1.2);
  }
  100% {
    transform: scale(1);
  }
}

/* Connection Arrow */
.stage-arrow {
  font-size: 2rem;
  color: #999;
  font-weight: bold;
  margin: 0 0.5rem;
  transition: color 0.5s ease;
}

.stage-item:has(.status-completed) + .stage-item .stage-arrow {
  color: #666;
}

/* Output Image */
.output-image-container {
  position: absolute;
  transform: translate(0, -50%);
  width: 15vw;
  max-width: 200px;
  min-width: 150px;
  aspect-ratio: 1 / 1;
  cursor: pointer;
  transition: all 0.3s ease;
  animation: image-appear 0.5s ease;
}

@keyframes image-appear {
  from {
    opacity: 0;
    transform: translate(0, -50%) scale(0.8);
  }
  to {
    opacity: 1;
    transform: translate(0, -50%) scale(1);
  }
}

.output-image-container:hover {
  transform: translate(0, -50%) scale(1.05);
}

.output-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 50%;
  box-shadow: 0 6px 30px rgba(76, 175, 80, 0.4);
  border: 3px solid #4CAF50;
}

.image-label {
  position: absolute;
  bottom: -2rem;
  left: 50%;
  transform: translateX(-50%);
  font-weight: 700;
  font-size: 0.8rem;
  color: #4CAF50;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
</style>
