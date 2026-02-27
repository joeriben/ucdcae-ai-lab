import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

export const useUiModeStore = defineStore('uiMode', () => {
  const mode = ref<'kids' | 'youth' | 'expert'>('youth') // safe default
  const loaded = ref(false)

  async function fetchMode() {
    try {
      const baseUrl = import.meta.env.DEV ? 'http://localhost:17802' : ''
      const { data } = await axios.get(`${baseUrl}/api/settings/ui-mode`)
      mode.value = data.ui_mode
      loaded.value = true
    } catch (e) {
      console.error('[UiMode] Failed to fetch UI mode:', e)
      loaded.value = true
    }
  }

  const isExpert = computed(() => mode.value === 'expert')
  const isKids = computed(() => mode.value === 'kids')

  return { mode, loaded, isExpert, isKids, fetchMode }
})
