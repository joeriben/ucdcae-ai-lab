<template>
  <div dir="ltr" class="denoising-progress-view">
    <!-- Phase A: Model Loading (no preview yet) -->
    <div v-if="!previewImage" class="model-loading-phase">
      <div class="model-card">
        <div class="model-card-header">
          <span class="model-icon">{{ profileIcon }}</span>
          <span class="model-name">{{ profileName }}</span>
          <span class="model-card-label">{{ t('edutainment.denoising.modelCard') }}</span>
        </div>

        <div class="model-specs">
          <div v-if="profile.publisher" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.publisher') }}</span>
            <span class="spec-value">{{ profile.publisher }}</span>
          </div>
          <div v-if="profile.architecture" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.architecture') }}</span>
            <span class="spec-value">{{ profile.architecture }}</span>
          </div>
          <div v-if="profile.params" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.parameters') }}</span>
            <span class="spec-value">{{ profile.params }}</span>
          </div>
          <div v-if="profile.textEncoders && profile.textEncoders.length" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.textEncoders') }}</span>
            <span class="spec-value">{{ profile.textEncoders.join(', ') }}</span>
          </div>
          <div v-if="profile.quantization" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.quantization') }}</span>
            <span class="spec-value">{{ profile.quantization }}</span>
          </div>
          <div v-if="vramDisplay" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.vramRequired') }}</span>
            <span class="spec-value">{{ vramDisplay }}</span>
          </div>
          <div v-if="resolutionDisplay" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.resolution') }}</span>
            <span class="spec-value">{{ resolutionDisplay }}</span>
          </div>
          <div v-if="profile.license" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.license') }}</span>
            <span class="spec-value">{{ profile.license }}</span>
          </div>
          <div v-if="profile.fairCulture" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.fairCulture') }}</span>
            <span class="spec-value">{{ profile.fairCulture }}</span>
          </div>
          <div v-if="profile.safetyByDesign" class="spec-row">
            <span class="spec-label">{{ t('edutainment.denoising.safetyByDesign') }}</span>
            <span class="spec-value">{{ profile.safetyByDesign }}</span>
          </div>
        </div>
      </div>

      <div class="loading-bar-section">
        <div class="loading-bar-track">
          <div class="loading-bar-fill loading-bar-indeterminate"></div>
        </div>
        <span class="loading-label">{{ t('edutainment.denoising.modelLoading') }}</span>
      </div>

      <!-- Rotating fact -->
      <div v-if="currentFact" class="expert-fact">
        {{ currentFact.text }}
      </div>
    </div>

    <!-- Phase B: Denoising Active (preview available) -->
    <div v-else class="denoising-active-phase">
      <div class="preview-container">
        <img :src="previewImage" alt="" class="denoising-preview-large" />
      </div>

      <div class="denoising-stats">
        <div class="step-progress">
          <span class="step-text">
            {{ stepDisplay }}
          </span>
          <div class="step-bar-track">
            <div class="step-bar-fill" :style="{ width: progress + '%' }"></div>
          </div>
          <span class="step-percent">{{ Math.round(progress) }}%</span>
        </div>

        <div class="stats-line">
          <span class="stats-model">{{ profileName }}</span>
          <span v-if="gpuStats.available && gpuStats.power_draw_watts" class="stats-gpu">
            {{ Math.round(gpuStats.power_draw_watts) }}W
          </span>
          <span v-if="gpuStats.available && gpuStats.memory_used_mb && gpuStats.memory_total_mb" class="stats-vram">
            {{ (gpuStats.memory_used_mb / 1024).toFixed(1) }}/{{ (gpuStats.memory_total_mb / 1024).toFixed(0) }} GB VRAM
          </span>
        </div>
      </div>

      <!-- Rotating fact -->
      <div v-if="currentFact" class="expert-fact">
        {{ currentFact.text }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useEdutainmentFacts } from '@/composables/useEdutainmentFacts'

const { t } = useI18n()

interface Props {
  progress: number
  previewImage: string | null
  modelMeta: Record<string, any> | null
  estimatedSeconds?: number
}

const props = withDefaults(defineProps<Props>(), {
  progress: 0,
  previewImage: null,
  modelMeta: null,
  estimatedSeconds: 30
})

// --- Model Profiles (static lookup) ---
interface ModelProfile {
  publisher: string
  architecture: string
  params: string
  textEncoders?: string[]
  quantization?: string
  license?: string
  fairCulture?: string
  safetyByDesign?: string
}

// Sources for model profile data:
// Flux 2 Dev:  https://huggingface.co/black-forest-labs/FLUX.2-dev
//              https://github.com/black-forest-labs/flux2
//              https://deepwiki.com/black-forest-labs/flux2/3.1-text-encoder-(mistral-small-3.2-24b)
// SD 3.5:      https://huggingface.co/stabilityai/stable-diffusion-3.5-large
//              https://stability.ai/news/introducing-stable-diffusion-3-5
// Wan 2.2:     https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
//              https://github.com/Wan-Video/Wan2.2
// GPT-Image-1: https://platform.openai.com/docs/guides/image-generation
// HeartMuLa:   https://arxiv.org/abs/2601.10547
//              https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B
// StableAudio:  https://huggingface.co/stabilityai/stable-audio-open-1.0
//              https://arxiv.org/html/2407.14358v1
//              Training data attribution: https://info.stability.ai/attributions
const MODEL_PROFILES: Record<string, ModelProfile> = {
  'flux2': {
    // 32B total (24B Mistral encoder + 8B DiT). Released Nov 2025.
    publisher: 'Black Forest Labs',
    architecture: 'Rectified Flow + MMDiT (32B)',
    params: '~32B',
    textEncoders: ['Mistral-Small-3.2-24B'],
    quantization: 'FP8 mixed precision',
    license: 'Non-Commercial (BFL Dev License)',
    fairCulture: 'Training data undisclosed, NSFW-filtered',
    safetyByDesign: 'NSFW filter + IWF CSAM filtering',
  },
  'sd35_large': {
    // 8.1B MMDiT, triple text encoder
    publisher: 'Stability AI',
    architecture: 'Rectified Flow + MMDiT',
    params: '~8.1B',
    textEncoders: ['CLIP-L', 'OpenCLIP-G', 'T5-XXL'],
    quantization: 'FP16',
    license: 'Stability Community License (<$1M)',
    fairCulture: 'Training data undisclosed ("synthetic + filtered public")',
    safetyByDesign: 'Filtered training data, structured red-teaming',
  },
  'wan22_t2v': {
    // 5B dense TI2V transformer (Wan 2.2, not 2.1)
    publisher: 'Wan-AI (Alibaba)',
    architecture: 'Diffusion Transformer (TI2V, dense)',
    params: '~5B',
    textEncoders: ['uMT5 (multilingual)'],
    quantization: 'BF16 (VAE: FP32)',
    license: 'Apache 2.0',
    fairCulture: 'Trained on ~1.5B videos + 10B images (sources undisclosed)',
    safetyByDesign: 'Optional safety checker in pipeline',
  },
  'gpt_image_1': {
    publisher: 'OpenAI',
    architecture: 'Autoregressive + Diffusion',
    params: 'Undisclosed',
    license: 'Commercial API (usage-based)',
    fairCulture: 'Training data undisclosed',
    safetyByDesign: 'Integrated content moderation, C2PA metadata',
  },
  'heartmula': {
    // arxiv 2601.10547, 3B released open-source
    publisher: 'HeartMuLa Research',
    architecture: 'Flow Matching + MuLa Codec (Global+Local Transformer)',
    params: '~3B',
    textEncoders: ['Internal tokenizer (lyrics + tags)'],
    quantization: 'FP16 / FP32 codec',
    license: 'Apache 2.0',
    fairCulture: '~600K songs + TTS corpora (song sources undisclosed)',
  },
  'stableaudio': {
    // Best provenance: 486K CC recordings, full attribution published
    publisher: 'Stability AI',
    architecture: 'Latent Diffusion + DiT (Audio)',
    params: '~1.2B',
    textEncoders: ['T5-Base (768d)'],
    quantization: 'FP32',
    license: 'Stability Community License (<$1M)',
    fairCulture: '486K CC recordings (Freesound + FMA), full attribution',
    safetyByDesign: 'PANNs + Audible Magic copyright filtering',
  },
  'surrealization': {
    // Custom: CLIP-L extrapolation on top of SD3.5
    publisher: 'AI4ArtsEd (Custom)',
    architecture: 'CLIP-L/T5 Extrapolation + SD3.5 MMDiT',
    params: '~8.1B (base) + extrapolation layer',
    textEncoders: ['CLIP-L (768d)', 'T5-XXL'],
    quantization: 'FP16',
    license: 'Research (SD3.5 Community License)',
    fairCulture: 'Based on SD3.5 training data (see above)',
    safetyByDesign: 'Platform 4-stage safety pipeline',
  },
}

/**
 * Match output_config name to a model profile key
 */
function resolveProfileKey(meta: Record<string, any> | null): string {
  if (!meta) return ''
  const modelFile = (meta.model_file || '') as string
  const backendType = (meta.backend_type || '') as string

  // Match by model_file first (most specific)
  if (modelFile.includes('flux2') || modelFile.includes('flux_dev')) return 'flux2'
  if (modelFile.includes('sd3.5') || modelFile.includes('sd35')) return 'sd35_large'

  // Match by backend_type
  if (backendType === 'heartmula') return 'heartmula'
  if (backendType === 'openai') return 'gpt_image_1'

  return ''
}

const profileKey = computed(() => resolveProfileKey(props.modelMeta))
const profile = computed<ModelProfile>(() => MODEL_PROFILES[profileKey.value] || {
  publisher: 'Unknown',
  architecture: props.modelMeta?.backend_type || 'Unknown',
  params: '',
})

const profileName = computed(() => {
  if (props.modelMeta?.model_file) {
    const f = props.modelMeta.model_file as string
    if (f.includes('flux2')) return 'FLUX.2 [dev]'
    if (f.includes('sd3.5')) return 'Stable Diffusion 3.5 Large'
  }
  if (props.modelMeta?.backend_type === 'heartmula') return 'HeartMuLa'
  if (props.modelMeta?.backend_type === 'openai') return 'GPT-Image-1'
  return props.modelMeta?.model_file || 'Model'
})

const profileIcon = computed(() => {
  const key = profileKey.value
  if (key === 'flux2') return '\u26A1'
  if (key === 'sd35_large') return '\uD83C\uDFA8'
  if (key === 'wan22_t2v') return '\uD83C\uDFAC'
  if (key === 'heartmula') return '\uD83C\uDFB5'
  if (key === 'stableaudio') return '\uD83C\uDFB6'
  if (key === 'gpt_image_1') return '\uD83E\uDDE0'
  if (key === 'surrealization') return '\uD83C\uDF00'
  return '\uD83E\uDD16'
})

const vramDisplay = computed(() => {
  const vram = props.modelMeta?.gpu_vram_mb
  if (!vram) return ''
  return `~${Math.round(vram / 1024)} GB`
})

const resolutionDisplay = computed(() => {
  return props.modelMeta?.recommended_resolution || ''
})

const stepDisplay = computed(() => {
  if (props.progress <= 0) return t('edutainment.denoising.denoisingActive')
  // Progress is 0-100, we don't have explicit step/total from SSE yet
  return t('edutainment.denoising.denoisingActive')
})

// --- Edutainment facts + GPU stats ---
const {
  gpuStats,
  currentFact,
  startRotation,
  stopRotation,
} = useEdutainmentFacts('expert')

// Start/stop rotation when component is shown
startRotation()
onUnmounted(() => stopRotation())

// Restart rotation if we transition between phases
watch(() => props.previewImage, () => {
  stopRotation()
  startRotation()
})
</script>

<style scoped>
.denoising-progress-view {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 0.5rem;
}

/* Phase A: Model Loading */
.model-loading-phase {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
}

.model-card {
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  padding: 1.25rem;
}

.model-card-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.model-icon {
  font-size: 1.5rem;
  line-height: 1;
}

.model-name {
  font-size: 1.1rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.95);
  flex: 1;
}

.model-card-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: rgba(76, 175, 80, 0.7);
  font-weight: 600;
}

.model-specs {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.4rem 1rem;
}

.spec-row {
  display: contents;
}

.spec-label {
  font-size: 0.78rem;
  color: rgba(255, 255, 255, 0.5);
  white-space: nowrap;
}

.spec-value {
  font-size: 0.78rem;
  color: rgba(255, 255, 255, 0.85);
  font-family: 'SF Mono', 'Fira Code', monospace;
}

/* Loading bar */
.loading-bar-section {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.loading-bar-track {
  height: 3px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 2px;
  overflow: hidden;
}

.loading-bar-fill {
  height: 100%;
  border-radius: 2px;
  background: rgba(76, 175, 80, 0.6);
}

.loading-bar-indeterminate {
  width: 40%;
  animation: indeterminate 1.8s ease-in-out infinite;
}

@keyframes indeterminate {
  0% { transform: translateX(-100%); }
  50% { transform: translateX(150%); }
  100% { transform: translateX(350%); }
}

.loading-label {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.45);
  text-align: center;
}

/* Phase B: Denoising Active */
.denoising-active-phase {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  width: 100%;
}

.preview-container {
  width: 100%;
  display: flex;
  justify-content: center;
}

.denoising-preview-large {
  max-width: 100%;
  max-height: clamp(300px, 40vh, 500px);
  object-fit: contain;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.denoising-stats {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.step-progress {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.step-text {
  font-size: 0.78rem;
  color: rgba(255, 255, 255, 0.6);
  white-space: nowrap;
  min-width: fit-content;
}

.step-bar-track {
  flex: 1;
  height: 4px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 2px;
  overflow: hidden;
}

.step-bar-fill {
  height: 100%;
  background: rgba(76, 175, 80, 0.7);
  border-radius: 2px;
  transition: width 0.3s ease;
}

.step-percent {
  font-size: 0.78rem;
  color: rgba(76, 175, 80, 0.8);
  font-weight: 600;
  font-family: 'SF Mono', 'Fira Code', monospace;
  min-width: 36px;
  text-align: right;
}

.stats-line {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.72rem;
  color: rgba(255, 255, 255, 0.4);
}

.stats-model {
  color: rgba(255, 255, 255, 0.6);
}

.stats-gpu,
.stats-vram {
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.stats-gpu::before,
.stats-vram::before {
  content: '\00B7';
  margin-right: 0.75rem;
  opacity: 0.4;
}

/* Expert fact (shared between phases) */
.expert-fact {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
  padding: 0.6rem 0.8rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
  border-left: 2px solid rgba(76, 175, 80, 0.3);
  line-height: 1.4;
  min-height: 2.5rem;
  transition: opacity 0.3s ease;
}

/* Responsive */
@media (max-width: 768px) {
  .model-specs {
    grid-template-columns: 1fr;
    gap: 0.25rem;
  }

  .spec-row {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }

  .denoising-preview-large {
    max-height: clamp(200px, 30vh, 350px);
  }
}
</style>
