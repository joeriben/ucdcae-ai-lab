<template>
  <div dir="ltr" class="iceberg-drawing-canvas" ref="containerRef">
    <canvas
      ref="canvasRef"
      :width="canvasWidth"
      :height="canvasHeight"
      @pointerdown="startDrawing"
      @pointermove="continueDrawing"
      @pointerup="finishDrawing"
      @pointerleave="finishDrawing"
      :class="{ drawing: isDrawing }"
    />

    <!-- Instructions overlay -->
    <div v-if="!hasDrawn && !isDrawing" class="instructions">
      <span class="instruction-text">{{ t('edutainment.iceberg.drawPrompt') }}</span>
    </div>

    <!-- Clear button -->
    <button
      v-if="hasDrawn && !disabled"
      class="clear-btn"
      @click="clearCanvas"
    >
      {{ t('edutainment.iceberg.redraw') }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { simplifyPolygon, type Point } from '@/composables/useIcebergPhysics'

const { t } = useI18n()

const props = defineProps<{
  disabled?: boolean
  icebergColor?: string
  waterLine?: number // Y position of water line (percentage from top)
}>()

const emit = defineEmits<{
  (e: 'iceberg-drawn', polygon: Point[]): void
  (e: 'drawing-cleared'): void
}>()

// Refs
const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const canvasWidth = ref(400)
const canvasHeight = ref(300)

// Drawing state
const isDrawing = ref(false)
const hasDrawn = ref(false)
const currentPath = ref<Point[]>([])

// Colors
const icebergFill = props.icebergColor || 'rgba(200, 230, 255, 0.9)'
const icebergStroke = '#4fa8d5'
const drawingStroke = '#ffffff'

/**
 * Start drawing on pointer down
 */
function startDrawing(event: PointerEvent) {
  if (props.disabled || hasDrawn.value) return

  isDrawing.value = true
  currentPath.value = []

  const point = getCanvasPoint(event)
  currentPath.value.push(point)

  // Capture pointer for smooth drawing
  ;(event.target as HTMLCanvasElement).setPointerCapture(event.pointerId)
}

/**
 * Continue drawing on pointer move
 */
function continueDrawing(event: PointerEvent) {
  if (!isDrawing.value || props.disabled) return

  const point = getCanvasPoint(event)

  // Only add point if it's far enough from the last one (reduces noise)
  const lastPoint = currentPath.value[currentPath.value.length - 1]
  if (!lastPoint) {
    currentPath.value.push(point)
    return
  }

  const distance = Math.sqrt(
    (point.x - lastPoint.x) ** 2 + (point.y - lastPoint.y) ** 2
  )

  if (distance > 3) {
    currentPath.value.push(point)
    drawCurrentPath()
  }
}

/**
 * Finish drawing on pointer up
 */
function finishDrawing(event: PointerEvent) {
  if (!isDrawing.value) return

  isDrawing.value = false

  // Release pointer capture
  ;(event.target as HTMLCanvasElement).releasePointerCapture(event.pointerId)

  // Only emit if we have enough points for a valid shape
  if (currentPath.value.length >= 5) {
    // Simplify the polygon to reduce points
    const simplified = simplifyPolygon(currentPath.value, 3)

    if (simplified.length >= 3) {
      hasDrawn.value = true
      drawFinalIceberg(simplified)
      emit('iceberg-drawn', simplified)
    }
  } else {
    // Not enough points - clear and let user try again
    clearCanvas()
  }
}

/**
 * Get canvas coordinates from pointer event
 */
function getCanvasPoint(event: PointerEvent): Point {
  const canvas = canvasRef.value
  if (!canvas) return { x: 0, y: 0 }

  const rect = canvas.getBoundingClientRect()
  const scaleX = canvas.width / rect.width
  const scaleY = canvas.height / rect.height

  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY
  }
}

/**
 * Draw the current path while user is drawing
 */
function drawCurrentPath() {
  const canvas = canvasRef.value
  const ctx = canvas?.getContext('2d')
  if (!ctx || !canvas) return

  // Clear and redraw
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  drawWaterLine(ctx)

  if (currentPath.value.length < 2) return

  const firstPoint = currentPath.value[0]
  if (!firstPoint) return

  // Draw the path
  ctx.beginPath()
  ctx.moveTo(firstPoint.x, firstPoint.y)

  for (let i = 1; i < currentPath.value.length; i++) {
    const point = currentPath.value[i]
    if (point) {
      ctx.lineTo(point.x, point.y)
    }
  }

  ctx.strokeStyle = drawingStroke
  ctx.lineWidth = 2
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.stroke()

  // Draw start point indicator
  ctx.beginPath()
  ctx.arc(firstPoint.x, firstPoint.y, 5, 0, Math.PI * 2)
  ctx.fillStyle = drawingStroke
  ctx.fill()
}

/**
 * Draw the final iceberg shape
 */
function drawFinalIceberg(polygon: Point[]) {
  const canvas = canvasRef.value
  const ctx = canvas?.getContext('2d')
  if (!ctx || !canvas) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  drawWaterLine(ctx)
  drawIceberg(ctx, polygon)
}

/**
 * Draw iceberg polygon
 */
function drawIceberg(ctx: CanvasRenderingContext2D, polygon: Point[]) {
  if (polygon.length < 3) return

  const firstPoint = polygon[0]
  if (!firstPoint) return

  ctx.beginPath()
  ctx.moveTo(firstPoint.x, firstPoint.y)

  for (let i = 1; i < polygon.length; i++) {
    const point = polygon[i]
    if (point) {
      ctx.lineTo(point.x, point.y)
    }
  }

  ctx.closePath()

  // Fill with ice color
  ctx.fillStyle = icebergFill
  ctx.fill()

  // Stroke outline
  ctx.strokeStyle = icebergStroke
  ctx.lineWidth = 2
  ctx.stroke()
}

/**
 * Draw the water line indicator
 */
function drawWaterLine(ctx: CanvasRenderingContext2D) {
  const canvas = canvasRef.value
  if (!canvas) return

  const waterY = canvas.height * (props.waterLine || 0.6)

  // Draw water area
  ctx.fillStyle = 'rgba(0, 100, 200, 0.2)'
  ctx.fillRect(0, waterY, canvas.width, canvas.height - waterY)

  // Draw water line
  ctx.setLineDash([5, 5])
  ctx.beginPath()
  ctx.moveTo(0, waterY)
  ctx.lineTo(canvas.width, waterY)
  ctx.strokeStyle = 'rgba(100, 180, 255, 0.5)'
  ctx.lineWidth = 1
  ctx.stroke()
  ctx.setLineDash([])
}

/**
 * Clear the canvas and reset state
 */
function clearCanvas() {
  const canvas = canvasRef.value
  const ctx = canvas?.getContext('2d')
  if (!ctx || !canvas) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  drawWaterLine(ctx)

  currentPath.value = []
  hasDrawn.value = false
  emit('drawing-cleared')
}

/**
 * Update canvas to show existing iceberg (for melting animation)
 */
function updateIceberg(polygon: Point[]) {
  drawFinalIceberg(polygon)
}

/**
 * Resize canvas to fit container
 */
function resizeCanvas() {
  if (!containerRef.value) return

  const rect = containerRef.value.getBoundingClientRect()
  canvasWidth.value = Math.floor(rect.width)
  canvasHeight.value = Math.floor(rect.height)

  // Redraw after resize
  requestAnimationFrame(() => {
    const canvas = canvasRef.value
    const ctx = canvas?.getContext('2d')
    if (ctx && canvas) {
      drawWaterLine(ctx)
    }
  })
}

// Watch for external polygon updates
watch(() => props.disabled, (disabled) => {
  if (disabled && hasDrawn.value) {
    // Redraw when disabled (during melting)
  }
})

// Lifecycle
onMounted(() => {
  resizeCanvas()
  window.addEventListener('resize', resizeCanvas)
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeCanvas)
})

// Expose methods for parent component
defineExpose({
  updateIceberg,
  clearCanvas
})
</script>

<style scoped>
.iceberg-drawing-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 200px;
  background: transparent;
}

canvas {
  display: block;
  width: 100%;
  height: 100%;
  cursor: crosshair;
  touch-action: none; /* Prevent scrolling on touch devices */
}

canvas.drawing {
  cursor: none;
}

.instructions {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  pointer-events: none;
}

.instruction-text {
  color: rgba(255, 255, 255, 0.7);
  font-size: 16px;
  font-family: system-ui, sans-serif;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
  background: rgba(0, 0, 0, 0.3);
  padding: 10px 20px;
  border-radius: 8px;
}

.clear-btn {
  position: absolute;
  bottom: 10px;
  right: 10px;
  padding: 6px 12px;
  background: rgba(255, 100, 100, 0.8);
  border: none;
  border-radius: 4px;
  color: white;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.2s;
}

.clear-btn:hover {
  background: rgba(255, 80, 80, 1);
}
</style>
