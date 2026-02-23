import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import i18n from './i18n'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)
app.use(i18n)

// Initialize global user preferences and sync with i18n
// Must be done after pinia and i18n are installed
import { useUserPreferencesStore } from './stores/userPreferences'
import { useSafetyLevelStore } from './stores/safetyLevel'
import { getLanguageDir } from './i18n'
import { watch } from 'vue'

const userPreferences = useUserPreferencesStore()

// Fetch safety level for feature gating (non-blocking)
const safetyLevelStore = useSafetyLevelStore()
safetyLevelStore.fetchLevel()

// Initial sync with vue-i18n + document direction
i18n.global.locale.value = userPreferences.language
document.documentElement.setAttribute('lang', userPreferences.language)
document.documentElement.setAttribute('dir', getLanguageDir(userPreferences.language))

// Watch for language changes and sync with i18n + document direction
watch(
  () => userPreferences.language,
  (newLanguage) => {
    i18n.global.locale.value = newLanguage
    document.documentElement.setAttribute('lang', newLanguage)
    document.documentElement.setAttribute('dir', getLanguageDir(newLanguage))
    console.log(`[i18n] Language synced to: ${newLanguage} (dir: ${getLanguageDir(newLanguage)})`)
  }
)

app.mount('#app')
