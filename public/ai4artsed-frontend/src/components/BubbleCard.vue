<template>
  <div class="bubble-card" :class="{ filled, required }">
    <div class="bubble-header">
      <span class="bubble-icon">{{ icon }}</span>
      <span class="bubble-label">{{ label }}</span>
      <div v-if="actions && actions.length > 0" class="bubble-actions">
        <button
          v-for="action in actions"
          :key="action.icon"
          @click="action.handler"
          class="action-btn"
          :title="action.title"
        >
          {{ action.icon }}
        </button>
      </div>
    </div>
    <slot></slot>
  </div>
</template>

<script setup lang="ts">
interface Action {
  icon: string
  title: string
  handler: () => void
}

interface Props {
  icon: string
  label: string
  filled?: boolean
  required?: boolean
  actions?: Action[]
}

withDefaults(defineProps<Props>(), {
  filled: false,
  required: false,
  actions: () => []
})
</script>

<style scoped>
.bubble-card {
  background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
  border: 2px solid #404040;
  border-radius: 16px;
  padding: 1.5rem;
  transition: all 0.3s ease;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.bubble-card.filled {
  border-color: #4a9eff;
  box-shadow: 0 0 20px rgba(74, 158, 255, 0.3);
}

.bubble-card.required {
  border-color: #ff6b6b;
  animation: pulse-required 2s ease-in-out infinite;
}

@keyframes pulse-required {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}

.bubble-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.bubble-icon {
  font-size: 1.5rem;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
}

.bubble-label {
  flex: 1;
  font-size: 1rem;
  font-weight: 600;
  color: #e0e0e0;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

.bubble-actions {
  display: flex;
  gap: 0.5rem;
}

.action-btn {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 0.4rem 0.6rem;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #e0e0e0;
}

.action-btn:hover {
  background: rgba(74, 158, 255, 0.2);
  border-color: #4a9eff;
  transform: scale(1.05);
}

.action-btn:active {
  transform: scale(0.95);
}
</style>
