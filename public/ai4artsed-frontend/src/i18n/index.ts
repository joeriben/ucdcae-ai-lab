import { createI18n } from 'vue-i18n'
import { de } from './de'
import { en } from './en'
import { tr } from './tr'
import { ko } from './ko'
import { uk } from './uk'
import { fr } from './fr'

export const SUPPORTED_LANGUAGES = [
  { code: 'de', label: 'Deutsch' },
  { code: 'en', label: 'English' },
  { code: 'tr', label: 'Türkçe' },
  { code: 'ko', label: '한국어' },
  { code: 'uk', label: 'Українська' },
  { code: 'fr', label: 'Français' },
] as const

export type SupportedLanguage = typeof SUPPORTED_LANGUAGES[number]['code']

/** Localized string object — en is mandatory, all others optional with fallback */
export type LocalizedString = { en: string; [key: string]: string }

/** Resolve a localized string for the given locale, falling back to English */
export function localized(obj: Record<string, string>, locale: string): string {
  return obj[locale] || obj.en || ''
}

const messages = { de, en, tr, ko, uk, fr }

// Cast locale to SupportedLanguage so vue-i18n accepts all supported languages
// (ko/tr messages are intentionally partial — fallbackLocale handles gaps)
export default createI18n({
  legacy: false,
  locale: 'de' as SupportedLanguage,
  fallbackLocale: 'en',
  messages
})
