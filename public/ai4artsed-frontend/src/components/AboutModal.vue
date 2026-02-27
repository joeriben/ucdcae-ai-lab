<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="modelValue" class="modal-overlay" @click="closeModal">
        <div class="modal-container" @click.stop>
          <div class="modal-header">
            <h1>{{ $t('about.title') }}</h1>
            <button class="modal-close" @click="closeModal" :title="$t('common.back')">×</button>
          </div>

          <div class="modal-body">
            <section class="intro">
              <p>{{ $t('about.intro') }}</p>
            </section>

            <section>
              <h2>{{ $t('about.project.title') }}</h2>
              <p>{{ $t('about.project.description') }}</p>
              <p>{{ $t('about.project.paragraph2') }}</p>
              <p>{{ $t('about.project.paragraph3') }}</p>
              <p>{{ $t('about.project.paragraph4') }}</p>
              <p>
                {{ $t('about.project.moreInfo') }}
                <a href="https://kubi-meta.de/ai4artsed" target="_blank" rel="noopener noreferrer">
                  kubi-meta.de/ai4artsed
                </a>
              </p>
            </section>

            <section>
              <h2>{{ $t('about.subproject.title') }}</h2>
              <p>{{ $t('about.subproject.description') }}</p>
            </section>

            <section>
              <h2>{{ $t('about.team.title') }}</h2>
              <div class="team-member">
                <h3>{{ $t('about.team.projectLead') }}</h3>
                <p><strong>{{ $t('about.team.leadName') }}</strong></p>
                <p>{{ $t('about.team.leadInstitute') }}<br>
                {{ $t('about.team.leadChair') }}<br>
                {{ $t('about.team.leadUnesco') }}</p>
                <p>E-Mail: <a href="mailto:benjamin.joerissen@fau.de">benjamin.joerissen@fau.de</a></p>
              </div>
              <div class="team-member">
                <h3>{{ $t('about.team.researcher') }}</h3>
                <p><strong>{{ $t('about.team.researcherName') }}</strong></p>
                <p>{{ $t('about.team.researcherInstitute') }}<br>
                {{ $t('about.team.researcherChair') }}<br>
                {{ $t('about.team.researcherUnesco') }}</p>
                <p>E-Mail: <a href="mailto:ucdcae@fau.de">ucdcae@fau.de</a></p>
              </div>
            </section>

            <section class="funding-section">
              <h2>{{ $t('about.funding.title') }}</h2>
              <div class="funding-logo">
                <a href="https://www.bmfsfj.de/" target="_blank" rel="noopener noreferrer">
                  <img src="/logos/BMBFSFJ_logo.png" alt="BMBFSFJ" />
                </a>
              </div>
              <div class="project-logos">
                <a href="https://kubi-meta.de/ai4artsed" target="_blank" rel="noopener noreferrer" class="project-logo-link">
                  <img src="/logos/ai4artsed_wordmark.png" alt="AI4ArtsEd" class="project-wordmark-img" />
                  <img src="/logos/ai4artsed_logo.jpg" alt="" class="project-logo-img" />
                </a>
                <a href="https://comearts.eu" target="_blank" rel="noopener noreferrer">
                  <img src="/logos/comearts_logo.jpg" alt="COMeARTS" class="project-logo-img" />
                </a>
              </div>
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

.team-member {
  margin-top: 2rem;
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 6px;
}

.team-member h3 {
  font-size: 1.1rem;
  font-weight: 600;
  color: #4CAF50;
  margin-bottom: 0.5rem;
  margin-top: 0;
}

.team-member:first-of-type {
  margin-top: 1rem;
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

.project-logos {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2rem;
  margin-top: 1.5rem;
}

.project-logo-img {
  max-height: 60px;
  width: auto;
}

.project-logo-link {
  display: flex;
  align-items: center;
  text-decoration: none;
  gap: 0;
}

.project-wordmark-img {
  max-height: 60px;
  width: auto;
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
