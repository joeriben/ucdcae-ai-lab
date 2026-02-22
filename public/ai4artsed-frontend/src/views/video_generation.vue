<template>
  <div class="video-generation-view">

    <div class="main-flow" ref="mainContainerRef">

      <!-- Section 1: Prompt Input -->
      <section class="prompt-section">
        <MediaInputBox
          ref="inputBoxRef"
          icon="🎬"
          :label="$t('videoGeneration.promptLabel')"
          :placeholder="$t('videoGeneration.promptPlaceholder')"
          v-model:value="promptText"
          input-type="text"
          :rows="4"
          :is-filled="!!promptText"
          @copy="copyPrompt"
          @paste="pastePrompt"
          @clear="clearPrompt"
        />
      </section>

      <!-- Section 2: Model Selection -->
      <section v-if="promptText.trim()" class="config-section">
        <h2 class="section-title">{{ $t('videoGeneration.modelLabel') }}</h2>
        <div class="config-bubbles-container">
          <div class="config-bubbles-row">
            <div
              v-for="config in availableConfigs"
              :key="config.id"
              class="config-bubble"
              :class="{
                selected: selectedConfig === config.id,
                hovered: hoveredConfigId === config.id
              }"
              :style="{ '--bubble-color': config.color }"
              @click="selectModel(config.id)"
              @mouseenter="hoveredConfigId = config.id"
              @mouseleave="hoveredConfigId = null"
              role="button"
              :aria-pressed="selectedConfig === config.id"
              tabindex="0"
              @keydown.enter="selectModel(config.id)"
              @keydown.space.prevent="selectModel(config.id)"
            >
              <div class="bubble-emoji-medium">{{ config.emoji }}</div>

              <!-- Hover info overlay -->
              <div v-if="hoveredConfigId === config.id" class="bubble-hover-info">
                <div class="hover-info-name">{{ config.name }}</div>
                <div class="hover-info-meta">
                  <div class="meta-row">
                    <span class="meta-label">Qual.</span>
                    <span class="meta-value">
                      <span class="stars-filled">{{ '\u2605'.repeat(config.quality) }}</span><span class="stars-unfilled">{{ '\u2606'.repeat(5 - config.quality) }}</span>
                    </span>
                  </div>
                  <div class="meta-row">
                    <span class="meta-label">Speed</span>
                    <span class="meta-value">
                      <span class="stars-filled">{{ '\u2605'.repeat(config.speed) }}</span><span class="stars-unfilled">{{ '\u2606'.repeat(5 - config.speed) }}</span>
                    </span>
                  </div>
                  <div class="meta-row">
                    <span class="meta-value duration-only">{{ config.resolution }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- START BUTTON -->
      <div class="start-button-container">
        <button
          class="start-button"
          :class="{ disabled: !canStartGeneration || isPipelineExecuting }"
          :disabled="!canStartGeneration || isPipelineExecuting"
          @click="startGeneration"
        >
          <span class="button-arrows button-arrows-left">&gt;&gt;&gt;</span>
          <span class="button-text">{{ isPipelineExecuting ? $t('videoGeneration.generating') : 'Start' }}</span>
          <span class="button-arrows button-arrows-right">&gt;&gt;&gt;</span>
        </button>

        <!-- Safety Badges -->
        <SafetyBadges v-if="safetyChecks.length > 0" :checks="safetyChecks" />
      </div>

      <!-- OUTPUT BOX -->
      <MediaOutputBox
        ref="pipelineSectionRef"
        :output-image="outputMedia"
        :media-type="'video'"
        :is-executing="isPipelineExecuting"
        :progress="generationProgress"
        :estimated-seconds="estimatedGenerationSeconds"
        :is-analyzing="false"
        :show-analysis="false"
        :analysis-data="null"
        :run-id="currentRunId"
        :is-favorited="isFavorited"
        @save="saveMedia"
        @download="downloadMedia"
        @toggle-favorite="toggleFavorite"
      />

    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import MediaOutputBox from '@/components/MediaOutputBox.vue'
import MediaInputBox from '@/components/MediaInputBox.vue'
import SafetyBadges from '@/components/SafetyBadges.vue'
import { useFavoritesStore } from '@/stores/favorites'
import { useAppClipboard } from '@/composables/useAppClipboard'
import { useDeviceId } from '@/composables/useDeviceId'
import { getModelAvailability, type ModelAvailability } from '@/services/api'
import { usePageContextStore } from '@/stores/pageContext'
import { useGenerationStream } from '@/composables/useGenerationStream'
import type { PageContext, FocusHint } from '@/composables/usePageContext'

// ============================================================================
// STATE
// ============================================================================

const { copy: copyToClipboard, paste: pasteFromClipboard } = useAppClipboard()

const promptText = ref('')
const selectedConfig = ref<string | null>(null)
const hoveredConfigId = ref<string | null>(null)

// Model availability
const modelAvailability = ref<ModelAvailability>({})
const availabilityLoading = ref(true)

// Seed management
const previousPrompt = ref('')
const currentSeed = ref<number | null>(null)

const deviceId = useDeviceId()

// Execution
const isPipelineExecuting = ref(false)
const outputMedia = ref<string | null>(null)
const currentRunId = ref<string | null>(null)

// SSE streaming
const {
  safetyChecks,
  generationProgress,
  currentStage,
  executeWithStreaming,
  reset: resetGenerationStream
} = useGenerationStream()

// Refs
const mainContainerRef = ref<HTMLElement | null>(null)
const inputBoxRef = ref<any>(null)
const pipelineSectionRef = ref<any>(null)

// ============================================================================
// Page Context for Trashy
// ============================================================================
const pageContextStore = usePageContextStore()

const trashyFocusHint = computed<FocusHint>(() => {
  if (isPipelineExecuting.value || outputMedia.value) {
    return { x: 95, y: 85, anchor: 'bottom-right' }
  }
  if (selectedConfig.value) {
    return { x: 95, y: 70, anchor: 'bottom-right' }
  }
  return { x: 2, y: 95, anchor: 'bottom-left' }
})

const pageContext = computed<PageContext>(() => ({
  activeViewType: 'video_generation',
  pageContent: {
    promptText: promptText.value,
    selectedConfig: selectedConfig.value
  },
  focusHint: trashyFocusHint.value
}))

watch(pageContext, (ctx) => {
  pageContextStore.setPageContext(ctx)
}, { immediate: true, deep: true })

onUnmounted(() => {
  pageContextStore.clearContext()
})

// ============================================================================
// CONFIGURATION
// ============================================================================

interface ModelConfig {
  id: string
  label: string
  emoji: string
  name: string
  quality: number
  speed: number
  duration: string
  resolution: string
  color: string
}

const videoModels: ModelConfig[] = [
  {
    id: 'wan21_t2v_14b_diffusers',
    label: 'Wan 14B',
    emoji: '\uD83C\uDFAC',
    name: 'Wan 2.1 14B (720p)',
    quality: 5,
    speed: 2,
    duration: '240',
    resolution: '1280\u00D7720',
    color: '#9B59B6'
  },
  {
    id: 'wan21_t2v_1_3b_diffusers',
    label: 'Wan 1.3B',
    emoji: '\u26A1',
    name: 'Wan 2.1 1.3B (480p)',
    quality: 3,
    speed: 4,
    duration: '60',
    resolution: '848\u00D7480',
    color: '#E67E22'
  }
]

// ============================================================================
// COMPUTED
// ============================================================================

const availableConfigs = computed(() => {
  if (Object.keys(modelAvailability.value).length > 0) {
    return videoModels.filter(config => modelAvailability.value[config.id] === true)
  }
  return videoModels
})

const estimatedGenerationSeconds = computed(() => {
  if (!selectedConfig.value) return 60
  const config = videoModels.find(c => c.id === selectedConfig.value)
  return parseInt(config?.duration || '60') || 60
})

const canStartGeneration = computed(() => {
  return (
    promptText.value.trim().length > 0 &&
    selectedConfig.value &&
    !isPipelineExecuting.value
  )
})

const favoritesStore = useFavoritesStore()

const isFavorited = computed(() => {
  if (!currentRunId.value) return false
  return favoritesStore.isFavorited(currentRunId.value)
})

// ============================================================================
// CLIPBOARD ACTIONS
// ============================================================================

function copyPrompt() {
  copyToClipboard(promptText.value)
}

function pastePrompt() {
  promptText.value = pasteFromClipboard()
}

function clearPrompt() {
  promptText.value = ''
  sessionStorage.removeItem('videogen_prompt')
}

// ============================================================================
// MODEL SELECTION
// ============================================================================

function selectModel(modelId: string) {
  selectedConfig.value = modelId
}

// ============================================================================
// GENERATION
// ============================================================================

async function startGeneration() {
  if (!canStartGeneration.value) return

  isPipelineExecuting.value = true
  resetGenerationStream()
  outputMedia.value = null

  await nextTick()
  setTimeout(() => scrollDownOnly(pipelineSectionRef.value?.sectionRef, 'start'), 150)

  let durationSeconds = estimatedGenerationSeconds.value * 0.9
  const targetProgress = 98
  const updateInterval = 100
  const totalUpdates = (durationSeconds * 1000) / updateInterval
  const progressPerUpdate = targetProgress / totalUpdates
  let progressInterval: ReturnType<typeof setInterval> | null = null

  const stopWatcher = watch(currentStage, (stage) => {
    if (stage === 'stage4' && !progressInterval) {
      progressInterval = setInterval(() => {
        if (generationProgress.value < targetProgress) {
          generationProgress.value += progressPerUpdate
          if (generationProgress.value > targetProgress) {
            generationProgress.value = targetProgress
          }
        }
      }, updateInterval)
    }
  }, { immediate: true })

  // Seed logic
  const promptChanged = promptText.value !== previousPrompt.value
  if (promptChanged || currentSeed.value === null) {
    currentSeed.value = Math.floor(Math.random() * 1000000000)
  }
  previousPrompt.value = promptText.value

  try {
    const result = await executeWithStreaming({
      prompt: promptText.value,
      output_config: selectedConfig.value || '',
      seed: currentSeed.value,
      input_text: promptText.value,
      device_id: deviceId
    })

    if (progressInterval) clearInterval(progressInterval)
    stopWatcher()

    if (result.status === 'success' && result.media_output) {
      const runId = result.media_output.run_id || result.run_id
      const mediaType = result.media_output.media_type || 'video'
      const mediaIndex = result.media_output.index ?? 0

      if (runId) {
        currentRunId.value = runId
        outputMedia.value = `/api/media/${mediaType}/${runId}/${mediaIndex}`

        await nextTick()
        setTimeout(() => scrollDownOnly(pipelineSectionRef.value?.sectionRef, 'start'), 150)
      }
    } else if (result.status === 'blocked') {
      generationProgress.value = 0
    } else {
      generationProgress.value = 0
    }
  } catch (error: any) {
    if (progressInterval) clearInterval(progressInterval)
    stopWatcher()
    console.error('[VideoGen] Error:', error)
    generationProgress.value = 0
  } finally {
    isPipelineExecuting.value = false
  }
}

// ============================================================================
// OUTPUT ACTIONS
// ============================================================================

function saveMedia() {
  alert('Save function coming soon!')
}

async function toggleFavorite() {
  if (!currentRunId.value) return
  await favoritesStore.toggleFavorite(currentRunId.value, 'video', deviceId, 'anonymous', 'video-generation')
}

async function downloadMedia() {
  if (!outputMedia.value) return

  try {
    const response = await fetch(outputMedia.value)
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url

    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')
    a.download = `ai4artsed_video_${timestamp}.mp4`

    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    console.error('[VideoGen Download] Error:', error)
  }
}

// ============================================================================
// SCROLL HELPERS
// ============================================================================

function scrollDownOnly(element: HTMLElement | null, block: ScrollLogicalPosition = 'start') {
  if (!element) return
  const rect = element.getBoundingClientRect()
  const targetTop = block === 'start' ? rect.top : rect.bottom - window.innerHeight
  if (targetTop > 0) {
    element.scrollIntoView({ behavior: 'smooth', block })
  }
}

// ============================================================================
// LIFECYCLE
// ============================================================================

onMounted(async () => {
  try {
    const result = await getModelAvailability()
    if (result.status === 'success') {
      modelAvailability.value = result.availability
    }
  } catch (error) {
    console.error('[VideoGen] Model availability error:', error)
  } finally {
    availabilityLoading.value = false
  }

  const savedPrompt = sessionStorage.getItem('videogen_prompt')
  if (savedPrompt) {
    promptText.value = savedPrompt
  }
})

watch(promptText, (newVal) => {
  sessionStorage.setItem('videogen_prompt', newVal)
})
</script>

<style scoped>
/* ============================================================================
   Root Container
   ============================================================================ */

.video-generation-view {
  min-height: 100%;
  background: #0a0a0a;
  color: #ffffff;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  overflow-y: auto;
  overflow-x: hidden;
  padding-bottom: 120px;
}

/* ============================================================================
   Main Flow
   ============================================================================ */

.main-flow {
  max-width: clamp(320px, 90vw, 900px);
  width: 100%;
  padding: clamp(1rem, 3vw, 2rem);

  display: flex;
  flex-direction: column;
  align-items: center;
  gap: clamp(1rem, 3vh, 2rem);
}

/* ============================================================================
   Prompt Section
   ============================================================================ */

.prompt-section {
  width: 100%;
  display: flex;
  justify-content: center;
}

/* ============================================================================
   Model Selection
   ============================================================================ */

.config-section {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.section-title {
  font-size: clamp(0.9rem, 2vw, 1rem);
  font-weight: 500;
  color: rgba(255, 255, 255, 0.6);
  margin: 0;
}

.config-bubbles-container {
  width: 100%;
  display: flex;
  justify-content: center;
}

.config-bubbles-row {
  display: inline-flex;
  flex-direction: row;
  gap: clamp(0.75rem, 2vw, 1rem);
  justify-content: center;
  flex-wrap: wrap;
  max-width: fit-content;
}

.config-bubble {
  position: relative;
  z-index: 1;
  width: clamp(80px, 12vw, 100px);
  height: clamp(80px, 12vw, 100px);
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(30, 30, 30, 0.9);
  border: 3px solid var(--bubble-color, rgba(255, 255, 255, 0.3));
  border-radius: 50%;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  outline: none;
}

.config-bubble:hover,
.config-bubble.hovered {
  transform: scale(2.0);
  background: rgba(20, 20, 20, 0.9);
  box-shadow: 0 0 30px var(--bubble-color);
  z-index: 100;
}

.config-bubble.selected {
  transform: scale(1.1);
  background: var(--bubble-color);
  box-shadow: 0 0 30px var(--bubble-color);
  border-color: #ffffff;
}

.config-bubble:focus-visible {
  outline: 3px solid rgba(102, 126, 234, 0.8);
  outline-offset: 4px;
}

.bubble-emoji-medium {
  font-size: clamp(2.5rem, 5vw, 3.5rem);
  line-height: 1;
}

/* Hover info overlay */
.bubble-hover-info {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 18%;
  color: white;
  z-index: 10;
  pointer-events: none;
  gap: 0.3rem;
}

.hover-info-name {
  font-size: 0.5rem;
  font-weight: 600;
  text-align: center;
  line-height: 1.25;
  color: rgba(255, 255, 255, 0.95);
  max-width: 100%;
  word-wrap: break-word;
}

.hover-info-meta {
  display: flex;
  flex-direction: column;
  gap: 0;
  width: 100%;
}

.meta-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.18rem;
  width: 100%;
  line-height: 1;
  margin: 0;
  padding: 0;
}

.meta-label {
  font-size: 0.45rem;
  color: rgba(255, 255, 255, 0.75);
  font-weight: 400;
  text-align: left;
  flex-shrink: 0;
  flex-basis: 35%;
}

.meta-value {
  font-size: 0.65rem;
  font-weight: 500;
  text-align: right;
  white-space: nowrap;
  flex-shrink: 0;
  flex-basis: 60%;
}

.stars-filled {
  color: #FFD700;
}

.stars-unfilled {
  color: rgba(150, 150, 150, 0.5);
}

.meta-value.duration-only {
  width: 100%;
  text-align: center;
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.45rem;
  flex-basis: auto;
  margin-top: 0.25rem;
  line-height: 1;
}

/* Hide emoji when hovering */
.config-bubble.hovered .bubble-emoji-medium {
  opacity: 0;
  display: none;
}

/* ============================================================================
   Start Button
   ============================================================================ */

.start-button-container {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: clamp(1rem, 3vw, 2rem);
  flex-wrap: wrap;
}

.start-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: clamp(0.5rem, 1.5vw, 0.75rem);
  padding: clamp(0.75rem, 2vw, 1rem) clamp(1.5rem, 4vw, 2.5rem);
  font-size: clamp(1rem, 2.5vw, 1.2rem);
  font-weight: 700;
  background: #000000;
  color: #FFB300;
  border: 3px solid #FFB300;
  border-radius: 16px;
  cursor: pointer;
  box-shadow: 0 0 20px rgba(255, 179, 0, 0.4),
              0 4px 15px rgba(0, 0, 0, 0.5);
  text-shadow: 0 0 10px rgba(255, 179, 0, 0.6);
  transition: all 0.3s ease;
}

.button-arrows {
  font-size: clamp(0.9rem, 2vw, 1.1rem);
}

.button-arrows-left {
  animation: arrow-pulse-left 1.5s ease-in-out infinite;
}

.button-arrows-right {
  animation: arrow-pulse-right 1.5s ease-in-out infinite;
}

.button-text {
  font-size: clamp(1rem, 2.5vw, 1.2rem);
}

@keyframes arrow-pulse-left {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.2); }
}

@keyframes arrow-pulse-right {
  0%, 100% { opacity: 1; transform: scale(1.2); }
  50% { opacity: 0.4; transform: scale(1); }
}

.start-button:hover {
  transform: scale(1.05) translateY(-2px);
  box-shadow: 0 0 30px rgba(255, 179, 0, 0.6),
              0 6px 25px rgba(0, 0, 0, 0.6);
  border-color: #FF8F00;
}

.start-button:active {
  transform: scale(0.98);
}

.start-button.disabled,
.start-button:disabled {
  opacity: 0.3;
  cursor: not-allowed;
  pointer-events: none;
  filter: grayscale(0.8);
  box-shadow: none;
  text-shadow: none;
}

.start-button.disabled .button-arrows,
.start-button:disabled .button-arrows {
  animation: none;
  opacity: 0.3;
}
</style>
