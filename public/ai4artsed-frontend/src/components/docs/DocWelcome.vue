<template>
  <section class="info-section">
    <h3>{{ currentLanguage === 'de' ? 'Was ist das UCDCAE AI LAB?' : 'What is the UCDCAE AI LAB?' }}</h3>
    <p>{{ currentLanguage === 'de'
      ? 'Das UCDCAE AI LAB ist eine pädagogisch-künstlerische Experimentierplattform des UNESCO Chair in Digital Culture and Arts in Education. Sie wurde entwickelt, um den kritischen und kreativen Umgang mit generativer KI in der kulturell-ästhetischen Medienbildung zu erforschen.'
      : 'The UCDCAE AI LAB is a pedagogical-artistic experimentation platform of the UNESCO Chair in Digital Culture and Arts in Education. It was developed to explore critical and creative engagement with generative AI in cultural-aesthetic media education.' }}</p>
  </section>

  <section class="info-section">
    <h3>{{ currentLanguage === 'de' ? 'Warum diese Plattform?' : 'Why this platform?' }}</h3>
    <p>{{ currentLanguage === 'de'
      ? 'Generative KI-Modelle sind mächtige Werkzeuge, aber auch "Black Boxes". Wir wollen verstehen: Wie reagieren verschiedene Modelle auf unterschiedliche Eingaben? Was passiert, wenn wir nicht nur kurze, einfache Prompts eingeben, sondern ausführliche, differenzierte Beschreibungen? Wie können wir lernen, selbst zu verstehen worum es uns geht? Wie können wir unsere Bildidee aus vielen unterschiedlichen Blickwinkeln verstehen und verändern?'
      : 'Generative AI models are powerful tools, but also "black boxes". We want to understand: How do different models react to different inputs? What happens when we don\'t just enter short, simple prompts, but detailed, nuanced descriptions? How can we learn to truly understand what we want? How can we understand and change our image idea from many different perspectives?' }}</p>
  </section>

  <section class="info-section">
    <h3>{{ currentLanguage === 'de' ? 'Kulturelle Sensibilität und Biases' : 'Cultural Sensitivity and Biases' }}</h3>
    <p>{{ currentLanguage === 'de'
      ? 'Ein zentrales Projektziel ist die kritische Erkundung generativer KI als Akteur in kulturellen, künstlerischen und gesellschaftlichen Zusammenhängen. Generative Modelle sind nicht neutral – sie tragen kulturelle, soziale, genderbezogene und ethnische Biases in sich, die aus ihren Trainingsdaten stammen. Diese Verzerrungen müssen sichtbar gemacht und kritisch untersucht werden, um zu verstehen, um welche Art von Akteur es sich bei generativer KI eigentlich handelt.'
      : 'A central project goal is the critical exploration of generative AI as an actor in cultural, artistic, and societal contexts. Generative models are not neutral – they carry cultural, social, gender-related, and ethnic biases inherited from their training data. These distortions must be made visible and critically examined in order to understand what kind of actor generative AI actually is.' }}</p>
  </section>

  <section class="info-section">
    <h3>{{ currentLanguage === 'de' ? 'Was passiert mit meinen Eingaben?' : 'What happens to my inputs?' }}</h3>
    <p>{{ currentLanguage === 'de'
      ? 'Generierte Inhalte werden zu Forschungszwecken gespeichert – sie helfen uns, die Plattform zu verbessern. Hochgeladene Bilder werden nicht dauerhaft gespeichert. Es werden keine personenbezogenen Daten erfasst.'
      : 'Generated content is saved for research purposes – it helps us improve the platform. Uploaded images are not permanently stored. No personal data is collected.' }}</p>
  </section>

  <section class="info-section contact-welcome">
    <h3>{{ currentLanguage === 'de' ? 'Kontakt' : 'Contact' }}</h3>
    <p>{{ currentLanguage === 'de'
      ? 'Frag Träshy direkt in der Anwendung, oder schreibe an: '
      : 'Ask Träshy directly in the application, or write to: ' }}<a href="mailto:ucdcae@fau.de">ucdcae@fau.de</a></p>
  </section>
</template>

<script setup lang="ts">
defineProps<{ currentLanguage: string }>()
</script>
