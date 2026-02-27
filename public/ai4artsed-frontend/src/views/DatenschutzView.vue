<template>
  <div class="legal-page">
    <div class="legal-container">
      <h1>{{ $t('legal.privacy.title') }}</h1>

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
</template>

<style scoped>
.legal-page {
  min-height: 100vh;
  background: #0a0a0a;
  padding: 2rem;
  color: #ffffff;
}

.legal-container {
  max-width: 900px;
  margin: 0 auto;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 2.5rem;
}

h1 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #ffffff;
  border-bottom: 2px solid rgba(76, 175, 80, 0.3);
  padding-bottom: 0.5rem;
}

h2 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-top: 2rem;
  margin-bottom: 0.75rem;
  color: rgba(255, 255, 255, 0.9);
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

section {
  margin-bottom: 1.5rem;
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

@media (max-width: 768px) {
  .legal-page {
    padding: 1rem;
  }

  .legal-container {
    padding: 1.5rem;
  }

  h1 {
    font-size: 1.5rem;
  }
}
</style>
