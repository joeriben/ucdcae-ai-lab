import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

/**
 * Pinia Store for Footer Gallery Favorites
 *
 * Manages:
 * - Favorites list (persisted on backend)
 * - Gallery expand/collapse state
 * - Favorite toggle actions
 * - Restore data fetching for session restoration
 *
 * Session 127: Footer Gallery Implementation
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/** Favorite item from backend */
export interface FavoriteItem {
  run_id: string
  device_id?: string  // Session 145: Per-user favorites (optional for backward compat)
  added_at: string
  thumbnail_url: string
  media_type: 'image' | 'audio' | 'video' | 'music' | 'text' | '3d' | 'midi' | 'p5' | 'sonicpi'
  user_id: string
  user_note: string
  source_view?: string  // Vue route path, e.g. 'surrealizer', 'text-transformation'
  // Enriched by backend
  exists?: boolean
  schema?: string
  timestamp?: string
  input_preview?: string
}

/** Response from GET /api/favorites */
interface FavoritesResponse {
  favorites: FavoriteItem[]
  total: number
  mode: 'global' | 'per_user'
}

/** Response from POST /api/favorites */
interface AddFavoriteResponse {
  success: boolean
  favorite: FavoriteItem
  total: number
}

/** Response from DELETE /api/favorites/{run_id} */
interface RemoveFavoriteResponse {
  success: boolean
  run_id: string
  total: number
}

/** Media output data for restore */
export interface MediaOutput {
  type: string
  filename: string
  url: string
  metadata: Record<string, unknown>
}

/** Models used at each pipeline stage */
export interface ModelsUsed {
  stage1_safety?: string
  stage2_interception?: string
  stage3_translation?: string
  stage3_safety?: string
  stage4_output?: string
}

/** Response from GET /api/favorites/{run_id}/restore */
export interface RestoreData {
  run_id: string
  schema: string
  timestamp: string
  current_state: Record<string, unknown>
  expected_outputs: string[]
  user_id: string
  input_text?: string
  context_prompt?: string  // Meta-Prompt/Regeln (user-editable!)
  transformed_text?: string
  translation_en?: string  // English translation for media generation
  models_used?: ModelsUsed  // LLM models used at each stage
  media_outputs: MediaOutput[]
  target_view: string
}

// ============================================================================
// STORE DEFINITION
// ============================================================================

export const useFavoritesStore = defineStore('favorites', () => {
  // ============================================================================
  // STATE
  // ============================================================================

  /** List of favorite items */
  const favorites = ref<FavoriteItem[]>([])

  /** Loading state */
  const isLoading = ref(false)

  /** Gallery expand/collapse state */
  const isGalleryExpanded = ref(false)

  /** Error message (if any) */
  const error = ref<string | null>(null)

  /** Favorites mode (global vs per_user) */
  const mode = ref<'global' | 'per_user'>('global')

  /** View mode: 'per_user' shows only own favorites, 'global' shows all */
  const viewMode = ref<'per_user' | 'global'>('per_user')  // Session 145: Default to per_user

  /** Pending restore data (set by FooterGallery, consumed by views via watcher) */
  const pendingRestoreData = ref<RestoreData | null>(null)

  // ============================================================================
  // COMPUTED
  // ============================================================================

  /** Total number of favorites */
  const totalFavorites = computed(() => favorites.value.length)

  /** Check if a run is favorited */
  const isFavorited = (runId: string): boolean => {
    return favorites.value.some((f) => f.run_id === runId)
  }

  /** Get favorites filtered by media type */
  const getFavoritesByType = (mediaType: string): FavoriteItem[] => {
    return favorites.value.filter((f) => f.media_type === mediaType)
  }

  /** Get only image favorites (for continue/img2img feature) */
  const imageFavorites = computed(() => {
    return favorites.value.filter((f) => f.media_type === 'image')
  })

  // ============================================================================
  // ACTIONS
  // ============================================================================

  /**
   * Load all favorites from backend
   *
   * @param deviceId - Optional device ID for filtering (per_user mode)
   */
  async function loadFavorites(deviceId?: string): Promise<void> {
    isLoading.value = true
    error.value = null

    try {
      console.log('[Favorites] Loading favorites from backend...')

      // Query parameters for filtering (Session 145)
      const params = new URLSearchParams()
      if (deviceId) {
        params.append('device_id', deviceId)
      }
      params.append('view_mode', viewMode.value)

      const response = await axios.get<FavoritesResponse>(`/api/favorites?${params.toString()}`)

      favorites.value = response.data.favorites
      mode.value = response.data.mode

      console.log(
        `[Favorites] Loaded ${response.data.total} favorites ` +
          `(backend mode: ${mode.value}, view mode: ${viewMode.value})`
      )
    } catch (e) {
      console.error('[Favorites] Failed to load favorites:', e)
      error.value = e instanceof Error ? e.message : 'Failed to load favorites'
      favorites.value = []
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Add a new favorite
   *
   * @param runId - Run ID to favorite
   * @param mediaType - Type of media
   * @param deviceId - Device identifier (browser_id + date) (Session 145)
   * @param userId - Optional user ID (default: 'anonymous')
   */
  async function addFavorite(
    runId: string,
    mediaType: FavoriteItem['media_type'],
    deviceId: string,
    userId: string = 'anonymous',
    sourceView?: string
  ): Promise<boolean> {
    error.value = null

    // Optimistic update
    const tempFavorite: FavoriteItem = {
      run_id: runId,
      device_id: deviceId,  // Session 145
      added_at: new Date().toISOString(),
      thumbnail_url: `/api/media/${mediaType}/${runId}${mediaType === 'image' ? '/0' : ''}`,
      media_type: mediaType,
      user_id: userId,
      user_note: '',
      source_view: sourceView,
      exists: true
    }

    // Add to beginning (most recent first)
    favorites.value.unshift(tempFavorite)

    try {
      console.log(`[Favorites] Adding favorite: ${runId} (${mediaType})`)
      const response = await axios.post<AddFavoriteResponse>('/api/favorites', {
        run_id: runId,
        media_type: mediaType,
        device_id: deviceId,  // Session 145
        user_id: userId,
        source_view: sourceView
      })

      // Update with server response
      const index = favorites.value.findIndex((f) => f.run_id === runId)
      if (index >= 0) {
        favorites.value[index] = response.data.favorite
      }

      console.log(`[Favorites] Successfully added favorite: ${runId}`)
      return true
    } catch (e) {
      // Rollback optimistic update
      favorites.value = favorites.value.filter((f) => f.run_id !== runId)

      if (axios.isAxiosError(e) && e.response?.status === 409) {
        // Already exists - not an error
        console.log(`[Favorites] Favorite already exists: ${runId}`)
        // Reload to get current state (Session 145: pass deviceId)
        await loadFavorites(deviceId)
        return true
      }

      console.error(`[Favorites] Failed to add favorite: ${runId}`, e)
      error.value = e instanceof Error ? e.message : 'Failed to add favorite'
      return false
    }
  }

  /**
   * Remove a favorite
   *
   * @param runId - Run ID to remove from favorites
   * @param deviceId - Optional device ID for ownership validation (Session 145)
   */
  async function removeFavorite(runId: string, deviceId?: string): Promise<boolean> {
    error.value = null

    // Store for potential rollback
    const removed = favorites.value.find((f) => f.run_id === runId)
    const originalIndex = favorites.value.findIndex((f) => f.run_id === runId)

    // Optimistic update
    favorites.value = favorites.value.filter((f) => f.run_id !== runId)

    try {
      console.log(`[Favorites] Removing favorite: ${runId}`)
      await axios.delete<RemoveFavoriteResponse>(`/api/favorites/${runId}`)

      console.log(`[Favorites] Successfully removed favorite: ${runId}`)
      return true
    } catch (e) {
      // Rollback optimistic update
      if (removed && originalIndex >= 0) {
        favorites.value.splice(originalIndex, 0, removed)
      }

      console.error(`[Favorites] Failed to remove favorite: ${runId}`, e)
      error.value = e instanceof Error ? e.message : 'Failed to remove favorite'
      return false
    }
  }

  /**
   * Toggle favorite status
   *
   * @param runId - Run ID to toggle
   * @param mediaType - Type of media (required for adding)
   * @param deviceId - Device identifier (Session 145)
   * @param userId - Optional user ID
   */
  async function toggleFavorite(
    runId: string,
    mediaType: FavoriteItem['media_type'],
    deviceId: string,
    userId: string = 'anonymous',
    sourceView?: string
  ): Promise<boolean> {
    if (isFavorited(runId)) {
      return removeFavorite(runId, deviceId)
    } else {
      return addFavorite(runId, mediaType, deviceId, userId, sourceView)
    }
  }

  /**
   * Get restore data for a favorite
   *
   * Returns complete session data for restoring UI state
   *
   * @param runId - Run ID to get restore data for
   */
  async function getRestoreData(runId: string): Promise<RestoreData | null> {
    try {
      console.log(`[Favorites] Getting restore data for: ${runId}`)
      const response = await axios.get<RestoreData>(`/api/favorites/${runId}/restore`)
      return response.data
    } catch (e) {
      console.error(`[Favorites] Failed to get restore data: ${runId}`, e)
      error.value = e instanceof Error ? e.message : 'Failed to get restore data'
      return null
    }
  }

  /**
   * Toggle gallery expanded state
   */
  function toggleGallery(): void {
    isGalleryExpanded.value = !isGalleryExpanded.value
    console.log(`[Favorites] Gallery ${isGalleryExpanded.value ? 'expanded' : 'collapsed'}`)
  }

  /**
   * Expand gallery
   */
  function expandGallery(): void {
    isGalleryExpanded.value = true
  }

  /**
   * Collapse gallery
   */
  function collapseGallery(): void {
    isGalleryExpanded.value = false
  }

  /**
   * Clear error
   */
  function clearError(): void {
    error.value = null
  }

  /**
   * Toggle between per_user and global view mode (Session 145)
   *
   * @param newMode - 'per_user' or 'global'
   * @param deviceId - Device ID for filtering (required in per_user mode)
   */
  async function setViewMode(
    newMode: 'per_user' | 'global',
    deviceId?: string
  ): Promise<boolean> {
    try {
      console.log(`[Favorites] Switching to ${newMode} view`)
      viewMode.value = newMode

      // Reload with new filter
      await loadFavorites(deviceId)

      return true
    } catch (e) {
      console.error('[Favorites] Failed to switch view:', e)
      error.value = e instanceof Error ? e.message : 'Failed to switch view'
      return false
    }
  }

  /**
   * Set restore data for cross-component communication
   *
   * Used by FooterGallery to signal views to restore state.
   * Views watch this and consume it immediately.
   *
   * @param data - Restore data or null to clear
   */
  function setRestoreData(data: RestoreData | null): void {
    pendingRestoreData.value = data
  }

  // ============================================================================
  // RETURN PUBLIC API
  // ============================================================================

  return {
    // State
    favorites,
    isLoading,
    isGalleryExpanded,
    error,
    mode,
    viewMode,  // Session 145: Per-user favorites
    pendingRestoreData,

    // Computed
    totalFavorites,
    imageFavorites,

    // Getters (functions)
    isFavorited,
    getFavoritesByType,

    // Actions
    loadFavorites,
    addFavorite,
    removeFavorite,
    toggleFavorite,
    getRestoreData,
    setViewMode,  // Session 145: Toggle per_user/global view
    toggleGallery,
    expandGallery,
    collapseGallery,
    clearError,
    setRestoreData
  }
})
