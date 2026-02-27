<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="modelValue" class="modal-overlay" @click="closeModal">
        <div class="modal-container" @click.stop>
          <div class="modal-header">
            <h1>{{ $t('legal.privacy.title') }}</h1>
            <button class="modal-close" @click="closeModal" :title="$t('common.back')">×</button>
          </div>

          <div class="modal-body">
            <p class="privacy-notice">{{ $t('legal.privacy.notice') }}</p>
            <p class="privacy-notice" style="margin-top: 1.5rem;">{{ $t('legal.privacy.usage') }} <a href="mailto:ucdcae@fau.de">ucdcae@fau.de</a></p>

            <section>
              <h2>{{ $t('legal.privacy.controller.title') }}</h2>
              <p>
                {{ $t('legal.privacy.controller.content') }}<br>
                E-Mail: <a href="mailto:ucdcae@fau.de">ucdcae@fau.de</a>
              </p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.dataProcessing.title') }}</h2>
              <p>{{ $t('legal.privacy.dataProcessing.content') }}</p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.deviceId.title') }}</h2>
              <p>{{ $t('legal.privacy.deviceId.content') }}</p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.uploads.title') }}</h2>
              <p>{{ $t('legal.privacy.uploads.content') }}</p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.cookies.title') }}</h2>
              <p>{{ $t('legal.privacy.cookies.content') }}</p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.externalServices.title') }}</h2>
              <p>{{ $t('legal.privacy.externalServices.content') }}</p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.legalBasis.title') }}</h2>
              <p>{{ $t('legal.privacy.legalBasis.content') }}</p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.retention.title') }}</h2>
              <p>{{ $t('legal.privacy.retention.content') }} <a href="mailto:ucdcae@fau.de">ucdcae@fau.de</a></p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.rights.title') }}</h2>
              <p>{{ $t('legal.privacy.rights.content') }} <a href="mailto:ucdcae@fau.de">ucdcae@fau.de</a></p>
            </section>

            <section>
              <h2>{{ $t('legal.privacy.fauReference.title') }}</h2>
              <p>
                {{ $t('legal.privacy.fauReference.content') }}<br>
                <a href="https://www.fau.de/datenschutz" target="_blank" rel="noopener noreferrer">
                  www.fau.de/datenschutz
                </a>
              </p>
            </section>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()

function closeModal() {
  emit('update:modelValue', false)
}

function handleEscape(event: KeyboardEvent) {
  if (event.key === 'Escape' && props.modelValue) {
    closeModal()
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleEscape)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleEscape)
})
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.85);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  padding: 1rem;
  overflow-y: auto;
}

.modal-container {
  background: #0a0a0a;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  max-width: 900px;
  width: 100%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.modal-header h1 {
  font-size: 1.75rem;
  font-weight: 700;
  color: #ffffff;
  margin: 0;
}

.modal-close {
  background: transparent;
  border: none;
  color: rgba(255, 255, 255, 0.6);
  font-size: 2.5rem;
  line-height: 1;
  cursor: pointer;
  padding: 0;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.3s ease;
}

.modal-close:hover {
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.9);
}

.modal-body {
  padding: 2rem;
  overflow-y: auto;
  flex: 1;
  color: #ffffff;
}

.privacy-notice {
  font-size: 1.1rem;
  line-height: 1.8;
  color: rgba(255, 255, 255, 0.9);
  padding: 1.5rem;
  background: rgba(76, 175, 80, 0.1);
  border-left: 3px solid #4CAF50;
  border-radius: 4px;
}

h2 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-top: 2rem;
  margin-bottom: 0.75rem;
  color: rgba(255, 255, 255, 0.9);
}

section {
  margin-bottom: 1.5rem;
}

section:first-child h2 {
  margin-top: 0;
}

p {
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.8);
}

a {
  color: #4CAF50;
  text-decoration: none;
  transition: color 0.3s ease;
}

a:hover {
  color: #66BB6A;
  text-decoration: underline;
}

.funding-section {
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.funding-logo {
  margin-top: 1rem;
  text-align: center;
}

.funding-logo img {
  max-width: 400px;
  width: 100%;
  height: auto;
  background: white;
  padding: 1rem;
  border-radius: 8px;
}

/* Modal Fade Transition */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.3s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-active .modal-container,
.modal-fade-leave-active .modal-container {
  transition: transform 0.3s ease;
}

.modal-fade-enter-from .modal-container,
.modal-fade-leave-to .modal-container {
  transform: scale(0.95);
}

@media (max-width: 768px) {
  .modal-container {
    max-height: 95vh;
  }

  .modal-header {
    padding: 1rem 1.5rem;
  }

  .modal-header h1 {
    font-size: 1.5rem;
  }

  .modal-body {
    padding: 1.5rem;
  }

  .funding-logo img {
    max-width: 100%;
  }
}
</style>
