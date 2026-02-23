<template>
  <div dir="ltr" class="config-canvas">
    <!-- Config tiles -->
    <ConfigTile
      v-for="config in positionedConfigs"
      :key="config.id"
      :config="config"
      :x="config.position!.x"
      :y="config.position!.y"
      :is-dimmed="isDimmed"
      :selected-properties="selectedProperties"
      :current-language="currentLanguage"
      @select="handleConfigSelect"
    />

    <!-- Match counter -->
    <div v-if="!isDimmed" class="match-counter">
      {{ matchCount }} {{ matchCount === 1 ? 'config' : 'configs' }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import ConfigTile from './ConfigTile.vue'
import type { ConfigMetadata } from '@/stores/configSelection'

/**
 * ConfigCanvas - Quadrants I, III, IV (upper-right, lower-left, lower-right)
 *
 * Displays config tiles with random distribution and overlap prevention.
 * Uses grid+jitter algorithm for positioning.
 *
 * Session 35 - Phase 1 Property Quadrants Implementation
 */

interface ConfigWithPosition extends ConfigMetadata {
  position?: { x: number; y: number }
}

interface Props {
  configs: ConfigMetadata[]
  selectedProperties: string[]
  matchCount: number
  canvasWidth: number
  canvasHeight: number
  isDimmed?: boolean
  currentLanguage: 'en' | 'de'
}

const props = withDefaults(defineProps<Props>(), {
  isDimmed: false
})

const emit = defineEmits<{
  selectConfig: [configId: string]
}>()

// Geometric constants (fixed for visual consistency)
const CATEGORY_BUBBLE_DIAMETER = 100
const CONFIG_BUBBLE_DIAMETER = 240

/**
 * Calculate responsive radius for category circle
 * Uses 25% of the smaller canvas dimension to ensure circles stay circular
 * and centered regardless of canvas size or aspect ratio
 * MUST match PropertyCanvas.vue calculation
 */
function getResponsiveRadius(): number {
  const smallerDimension = Math.min(props.canvasWidth, props.canvasHeight)
  return smallerDimension * 0.25  // 25% of smaller dimension
}

// Positioned configs
const positionedConfigs = ref<ConfigWithPosition[]>([])

/**
 * Get category bubble position (matches PropertyCanvas.vue logic)
 * Uses responsive radius calculation
 */
function getCategoryPosition(category: string): { x: number; y: number } {
  // Center of canvas (true geometric center)
  const centerX = props.canvasWidth / 2
  const centerY = props.canvasHeight / 2

  // Responsive circle radius: proportional to canvas size
  const radius = getResponsiveRadius()

  console.log('[ConfigCanvas] Getting category position for', category, {
    canvasWidth: props.canvasWidth,
    canvasHeight: props.canvasHeight,
    centerX,
    centerY,
    radius,
    smallerDimension: Math.min(props.canvasWidth, props.canvasHeight)
  })

  // Freestyle in center
  if (category === 'freestyle') {
    return { x: centerX, y: centerY }
  }

  // Other categories in X-formation around center
  const otherCategories = ['semantics', 'aesthetics', 'arts', 'heritage'].filter(c => {
    // Only include categories that exist in configs
    return props.configs.some(cfg => cfg.properties.includes(c))
  })

  const index = otherCategories.indexOf(category)
  if (index === -1) {
    // Fallback for unknown category
    return { x: centerX, y: centerY }
  }

  const angleStep = (2 * Math.PI) / otherCategories.length
  const angle = index * angleStep - Math.PI / 4 // -45° start (X-formation)

  const x = centerX + Math.cos(angle) * radius
  const y = centerY + Math.sin(angle) * radius

  return { x, y }
}

/**
 * Calculate config bubble positions clustered around their category bubbles
 * Uses circular arrangement around each category
 */
function calculatePositions() {
  const configs = [...props.configs]

  if (configs.length === 0) {
    positionedConfigs.value = []
    return
  }

  // Group configs by category
  const configsByCategory = new Map<string, ConfigMetadata[]>()
  configs.forEach(config => {
    const category = config.properties[0] || 'freestyle' // First property is category
    if (!configsByCategory.has(category)) {
      configsByCategory.set(category, [])
    }
    configsByCategory.get(category)!.push(config)
  })

  // Position configs around their category bubbles
  const positioned: ConfigWithPosition[] = []

  configsByCategory.forEach((categoryConfigs, category) => {
    const categoryPos = getCategoryPosition(category)

    // Calculate radius for config cluster around this category
    // Distance from category center to config center
    // Use proportional spacing based on canvas size (15% of smaller dimension)
    const smallerDimension = Math.min(props.canvasWidth, props.canvasHeight)
    const proportionalSpacing = smallerDimension * 0.15
    const clusterRadius = (CATEGORY_BUBBLE_DIAMETER / 2) + (CONFIG_BUBBLE_DIAMETER / 2) + proportionalSpacing

    // Arrange configs in circle around category bubble
    const numConfigs = categoryConfigs.length
    const angleStep = (2 * Math.PI) / numConfigs

    categoryConfigs.forEach((config, index) => {
      const angle = index * angleStep
      const x = categoryPos.x + Math.cos(angle) * clusterRadius
      const y = categoryPos.y + Math.sin(angle) * clusterRadius

      // Clamp to canvas bounds
      const clampedX = Math.max(CONFIG_BUBBLE_DIAMETER / 2, Math.min(props.canvasWidth - CONFIG_BUBBLE_DIAMETER / 2, x))
      const clampedY = Math.max(CONFIG_BUBBLE_DIAMETER / 2, Math.min(props.canvasHeight - CONFIG_BUBBLE_DIAMETER / 2, y))

      positioned.push({
        ...config,
        position: { x: clampedX, y: clampedY }
      })
    })
  })

  positionedConfigs.value = positioned
}

function handleConfigSelect(configId: string) {
  emit('selectConfig', configId)
}

// Recalculate positions when configs OR canvas size changes
watch(() => props.configs, () => {
  calculatePositions()
}, { deep: true })

watch([() => props.canvasWidth, () => props.canvasHeight], () => {
  calculatePositions()
})

onMounted(() => {
  calculatePositions()
})
</script>

<style scoped>
.config-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  pointer-events: none;
  z-index: 1;
}

/* Re-enable pointer events on actual tiles */
.config-canvas :deep(.config-tile) {
  pointer-events: all;
}

.match-counter {
  position: absolute;
  bottom: 20px;
  right: 20px;
  padding: 8px 16px;
  background: rgba(20, 20, 20, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 20px;
  color: rgba(255, 255, 255, 0.7);
  font-size: 13px;
  font-weight: 500;
  pointer-events: all;
  z-index: 100;
}
</style>
