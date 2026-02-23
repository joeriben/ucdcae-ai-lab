<template>
  <!-- Main Container - Fullscreen viewport overlay -->
  <div dir="ltr" class="property-canvas">
    <!-- Centered Container for all bubbles -->
    <div class="bubble-viewport">
      <!-- Inner Container - Maintains aspect ratio -->
      <div class="bubble-container">

        <!-- Category Bubbles - Always visible -->
        <div
          v-for="category in categories"
          :key="category.id"
          :class="[
            'category-bubble',
            `category-${category.id}`,
            { 'selected': selectedCategory === category.id }
          ]"
          :style="{
            '--bubble-color': category.color,
            left: category.position.x,
            top: category.position.y
          }"
          @click="toggleCategory(category.id)"
          role="button"
          :aria-pressed="selectedCategory === category.id"
          :aria-label="`Select ${category.name} category`"
        >
          <span class="bubble-symbol">{{ category.symbol }}</span>
          <span class="bubble-label">{{ category.name }}</span>
        </div>

        <!-- Configuration Bubbles - Only visible when category selected -->
        <transition-group
          name="config-fade"
          tag="div"
          class="config-bubbles-wrapper"
          v-if="selectedCategory && visibleConfigs.length > 0"
        >
          <div
            v-for="(config, index) in visibleConfigs"
            :key="config.id"
            class="config-bubble"
            :style="getConfigStyle(config, index)"
            @click="selectConfiguration(config)"
            role="button"
            :aria-label="`Select ${config.name} configuration`"
          >
            <div class="config-content">
              <img
                v-if="config.imageUrl"
                :src="config.imageUrl"
                :alt="config.name"
                loading="lazy"
              >
              <div v-else class="config-placeholder">
                {{ getConfigName(config).charAt(0) }}
              </div>
            </div>
            <span class="config-label">{{ getConfigName(config) }}</span>
          </div>
        </transition-group>

      </div>
    </div>

    <!-- Match Counter -->
    <div v-if="selectedCategory && visibleConfigs.length > 0" class="match-counter">
      {{ visibleConfigs.length }} {{ visibleConfigs.length === 1 ? 'Konfiguration' : 'Konfigurationen' }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useConfigSelectionStore } from '@/stores/configSelection'
import { useUserPreferencesStore } from '@/stores/userPreferences'

// Props
interface Props {
  selectedProperties?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  selectedProperties: () => []
})

// Emits
const emit = defineEmits<{
  toggleProperty: [property: string]
  selectConfig: [configId: string]
}>()

// Router and Stores
const router = useRouter()
const configStore = useConfigSelectionStore()
const userPreferences = useUserPreferencesStore()

// State Management
const selectedCategory = ref<string | null>(null)

// Get categories from store
const categories = computed(() => configStore.categories)

// Category color mapping
const categoryColorMap: Record<string, string> = {
  semantics: '#2196F3',   // 💬
  aesthetics: '#9C27B0',  // 🪄
  arts: '#E91E63',        // 🖌️
  heritage: '#4CAF50',    // 🌍
  freestyle: '#FFC107',   // 🫵
}

// Category positions (calculated dynamically)
type CategoryPosition = { x: string; y: string }
const categoryPositions = ref<Record<string, CategoryPosition>>({})

// Calculate category positions in X-formation
function calculateCategoryPositions() {
  const positions: Record<string, CategoryPosition> = {}

  const centerX = 50
  const centerY = 50
  const radiusPercent = 35

  const all = categories.value

  // Freestyle in center if present
  if (all.includes('freestyle')) {
    positions.freestyle = { x: `${centerX}%`, y: `${centerY}%` }
  }

  const others = all.filter((c) => c !== 'freestyle')
  if (others.length === 0) {
    categoryPositions.value = positions
    return
  }

  const angleStep = (2 * Math.PI) / others.length

  others.forEach((category, index) => {
    // Start at -45° for X-formation
    const angle = index * angleStep - Math.PI / 4

    const x = centerX + Math.cos(angle) * radiusPercent
    const y = centerY + Math.sin(angle) * radiusPercent

    positions[category] = { x: `${x}%`, y: `${y}%` }
  })

  categoryPositions.value = positions
}

// Get category data (symbol, color, position)
const getCategoryData = (categoryId: string) => {
  const symbolData = configStore.getSymbolData(categoryId)
  return {
    id: categoryId,
    name: symbolData?.label || categoryId,
    symbol: symbolData?.symbol || '',
    color: categoryColorMap[categoryId] || '#888888',
    position: categoryPositions.value[categoryId] || { x: '50%', y: '50%' }
  }
}

// Get current language
const currentLanguage = computed(() => userPreferences.language)

// Get configurations from store - use availableConfigs which is the actual property
const allConfigurations = computed(() => configStore.availableConfigs)

// Filter configs based on selected category
const visibleConfigs = computed(() => {
  if (!selectedCategory.value) return []

  // Filter configs that belong to the selected category
  return allConfigurations.value.filter(config => {
    // Check if config has the selected category in its properties
    return config.properties && config.properties.includes(selectedCategory.value)
  })
})

// Get localized config name
const getConfigName = (config: any) => {
  if (typeof config.name === 'string') {
    return config.name
  }
  return config.name[currentLanguage.value] || config.name.en || ''
}

// Calculate config bubble position around its parent category
const getConfigStyle = (config: any, index: number) => {
  const categoryPos = getCategoryPosition(selectedCategory.value)
  const numConfigs = visibleConfigs.value.length

  // Calculate angle for this config (evenly distributed around circle)
  const angleStep = (2 * Math.PI) / numConfigs
  const angle = index * angleStep - Math.PI / 2 // Start at top (-90°)

  // Calculate distance from category center (responsive)
  // Use 20% of container size as radius
  const distance = 20

  // Calculate position using trigonometry
  const x = parseFloat(categoryPos.x) + Math.cos(angle) * distance
  const y = parseFloat(categoryPos.y) + Math.sin(angle) * distance

  return {
    left: `${x}%`,
    top: `${y}%`
  }
}

// Methods
const toggleCategory = (categoryId: string) => {
  // XOR logic - only one category can be selected at a time
  if (selectedCategory.value === categoryId) {
    selectedCategory.value = null
  } else {
    selectedCategory.value = categoryId
  }

  // Emit the toggle event for compatibility
  emit('toggleProperty', categoryId)
}

const getCategoryPosition = (categoryId: string | null) => {
  if (!categoryId) return { x: '50%', y: '50%' }
  const category = categories.value.find(c => c.id === categoryId)
  return category ? category.position : { x: '50%', y: '50%' }
}

const selectConfiguration = (config: any) => {
  // Emit event for parent component
  emit('selectConfig', config.id)

  // Navigate to pipeline execution
  router.push({
    name: 'pipeline-execution',
    params: { configId: config.id }
  })
}

// Initialize
onMounted(() => {
  // Load configs from store if not already loaded
  if (allConfigurations.value.length === 0) {
    configStore.loadConfigs()
  }
})
</script>

<style scoped>
/* Reset and Base Styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

/* Main Container - Fullscreen overlay */
.property-canvas {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none; /* Allow clicks through to underlying elements */
  z-index: 10;
}

/* Viewport Container - Centers the bubble area */
.bubble-viewport {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  position: relative;
}

/* Bubble Container - Maintains aspect ratio and relative positioning */
.bubble-container {
  position: relative;
  width: min(70vw, 70vh);
  height: min(70vw, 70vh);
  aspect-ratio: 1 / 1;
}

/* Category Bubbles */
.category-bubble {
  position: absolute;
  width: 12%;
  aspect-ratio: 1 / 1;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(20, 20, 20, 0.9);
  border: 3px solid var(--bubble-color);
  border-radius: 50%;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  pointer-events: all; /* Re-enable pointer events for bubbles */
}

.category-bubble:hover {
  transform: translate(-50%, -50%) scale(1.1);
  background: rgba(30, 30, 30, 0.95);
  box-shadow: 0 0 30px var(--bubble-color);
}

.category-bubble.selected {
  background: var(--bubble-color);
  transform: translate(-50%, -50%) scale(1.15);
  box-shadow: 0 0 40px var(--bubble-color);
  color: #0a0a0a;
}

.bubble-symbol {
  font-size: clamp(1.5rem, 4vw, 2.5rem);
  margin-bottom: 0.25rem;
}

.bubble-label {
  font-size: clamp(0.7rem, 1.5vw, 0.9rem);
  color: white;
  font-weight: 500;
  text-align: center;
}

.category-bubble.selected .bubble-label {
  color: white;
  font-weight: 600;
}

/* Configuration Bubbles Wrapper */
.config-bubbles-wrapper {
  position: absolute;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

/* Configuration Bubbles */
.config-bubble {
  position: absolute;
  width: 18%;
  aspect-ratio: 1 / 1;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  pointer-events: all;
  transition: all 0.3s ease;
}

.config-bubble:hover {
  transform: translate(-50%, -50%) scale(1.1);
  z-index: 10;
}

.config-content {
  width: 80%;
  aspect-ratio: 1 / 1;
  border-radius: 50%;
  overflow: hidden;
  background: white;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  margin-bottom: 0.5rem;
}

.config-content img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.config-placeholder {
  font-size: 2rem;
  font-weight: bold;
  color: #666;
}

.config-label {
  font-size: clamp(0.8rem, 1.5vw, 1rem);
  color: white;
  font-weight: 500;
  text-align: center;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

/* Match Counter */
.match-counter {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  padding: 0.75rem 1.5rem;
  background: rgba(20, 20, 20, 0.9);
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 2rem;
  color: white;
  font-size: 0.9rem;
  font-weight: 500;
  pointer-events: none;
  z-index: 100;
}

/* Transitions */
.config-fade-enter-active,
.config-fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.config-fade-enter-from {
  opacity: 0;
  transform: translate(-50%, -50%) scale(0.8);
}

.config-fade-leave-to {
  opacity: 0;
  transform: translate(-50%, -50%) scale(0.8);
}

/* Responsive Design */
@media (max-width: 1024px) {
  .bubble-container {
    width: min(90vw, 90vh);
    height: min(90vw, 90vh);
  }

  .category-bubble {
    width: 15%;
  }

  .config-bubble {
    width: 22%;
  }
}

@media (max-width: 768px) {
  .canvas-header {
    padding: 1rem 1.5rem;
  }

  .clear-button {
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
  }

  .bubble-symbol {
    font-size: clamp(1.2rem, 3.5vw, 2rem);
  }
}

/* Accessibility */
@media (prefers-reduced-motion: reduce) {
  .category-bubble,
  .config-bubble,
  .clear-button {
    transition: none;
  }

  .config-fade-enter-active,
  .config-fade-leave-active {
    transition: none;
  }
}

/* Focus styles for keyboard navigation */
.category-bubble:focus-visible,
.config-bubble:focus-visible,
.clear-button:focus-visible {
  outline: 3px solid white;
  outline-offset: 2px;
}
</style>