/* ============================================================
   ECO-LOOP Simulator — Apple Sample Data
   Defines the 4 sample apples and their internal truth states
   ============================================================ */

const APPLE_SAMPLES = [
  {
    id: "sample-1",
    label: "Sample 01",
    displayName: "Fresh Apple",
    description: "Visibly healthy red apple",

    // What it LOOKS like from outside (controls SVG appearance)
    appearance: {
      bodyColor:   "#d9342b",      // deep red
      highlightColor: "#f56358",
      shadowColor: "#a01b14",
      hasSpots:    false,
      spotIntensity: 0
    },

    // What it ACTUALLY is inside (the hidden truth)
    truth: {
      class: "Fresh",           // Fresh / Early-Rot / Severe-Rot
      classCode: 0,
      pH: 4.25,
      internalDescription: "Healthy white-cream flesh, intact cells, no rot"
    },

    // Spectral signature anchor — used to generate readings later
    spectralProfile: "fresh"
  },

  {
    id: "sample-2",
    label: "Sample 02",
    displayName: "Early-Rot Apple",
    description: "Slight discoloration visible",

    appearance: {
      bodyColor:    "#c44832",
      highlightColor: "#e26849",
      shadowColor:  "#8b2d1c",
      hasSpots:     true,
      spotIntensity: 0.4
    },

    truth: {
      class: "Early-Rot",
      classCode: 1,
      pH: 3.55,
      internalDescription: "Mild internal browning, some cellular breakdown"
    },

    spectralProfile: "early"
  },

  {
    id: "sample-3",
    label: "Sample 03",
    displayName: "Rotten Apple",
    description: "Visible decay and discoloration",

    appearance: {
      bodyColor:    "#8b5a3c",
      highlightColor: "#a07050",
      shadowColor:  "#5a3a25",
      hasSpots:     true,
      spotIntensity: 1.0
    },

    truth: {
      class: "Severe-Rot",
      classCode: 2,
      pH: 2.95,
      internalDescription: "Severe internal decay, brown patches throughout"
    },

    spectralProfile: "severe"
  },

  {
    id: "sample-4",
    label: "Sample 04",
    displayName: "Mystery Apple",
    description: "Unknown condition — randomized",

    // appearance and truth get randomized when selected
    isRandom: true,
    appearance: null,
    truth: null,
    spectralProfile: null
  }
];

/* ============================================================
   Random apple generator (for Sample 04 — the mystery apple)
   ============================================================ */

function generateMysteryApple() {
  // The interesting twist: sometimes the apple looks fresh
  // but is rotten inside. This is your project's biggest claim!

  const scenarios = [
    // Scenario A — looks fresh, IS fresh
    {
      appearance: {
        bodyColor: "#d9342b", highlightColor: "#f56358",
        shadowColor: "#a01b14", hasSpots: false, spotIntensity: 0
      },
      truth: { class: "Fresh", classCode: 0, pH: 4.18,
               internalDescription: "Healthy throughout" },
      spectralProfile: "fresh"
    },
    // Scenario B — LOOKS FRESH but ROTTEN inside (the key demo case!)
    {
      appearance: {
        bodyColor: "#cf3a28", highlightColor: "#ee5a48",
        shadowColor: "#971810", hasSpots: false, spotIntensity: 0
      },
      truth: { class: "Severe-Rot", classCode: 2, pH: 2.88,
               internalDescription: "Hidden internal rot — invisible from outside" },
      spectralProfile: "severe"
    },
    // Scenario C — slight visible spots, Early-Rot inside
    {
      appearance: {
        bodyColor: "#c44832", highlightColor: "#e26849",
        shadowColor: "#8b2d1c", hasSpots: true, spotIntensity: 0.3
      },
      truth: { class: "Early-Rot", classCode: 1, pH: 3.62,
               internalDescription: "Mild internal browning" },
      spectralProfile: "early"
    },
    // Scenario D — visibly rotten and is rotten
    {
      appearance: {
        bodyColor: "#9c6440", highlightColor: "#b07a58",
        shadowColor: "#683f28", hasSpots: true, spotIntensity: 0.9
      },
      truth: { class: "Severe-Rot", classCode: 2, pH: 2.92,
               internalDescription: "Severe decay throughout" },
      spectralProfile: "severe"
    }
  ];

  return scenarios[Math.floor(Math.random() * scenarios.length)];
}


/* ============================================================
   SVG Apple Generator — Returns SVG string for any apple
   ============================================================ */

function getAppleSVG(appearance, size = 200) {
  const { bodyColor, highlightColor, shadowColor, hasSpots, spotIntensity } = appearance;

  // Generate spots if needed
  let spots = '';
  if (hasSpots && spotIntensity > 0) {
    const spotCount = Math.floor(spotIntensity * 8);
    for (let i = 0; i < spotCount; i++) {
      const cx = 60 + Math.random() * 80;
      const cy = 70 + Math.random() * 70;
      const r = 3 + Math.random() * (spotIntensity * 6);
      const opacity = 0.3 + spotIntensity * 0.5;
      spots += `<circle cx="${cx}" cy="${cy}" r="${r}"
                fill="#4a2818" opacity="${opacity}" />`;
    }
  }

  return `
    <svg width="${size}" height="${size}" viewBox="0 0 200 200"
         xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="appleGradient" cx="35%" cy="30%" r="70%">
          <stop offset="0%"   stop-color="${highlightColor}" />
          <stop offset="60%"  stop-color="${bodyColor}" />
          <stop offset="100%" stop-color="${shadowColor}" />
        </radialGradient>
        <radialGradient id="highlight" cx="35%" cy="30%" r="20%">
          <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.4" />
          <stop offset="100%" stop-color="#ffffff" stop-opacity="0" />
        </radialGradient>
      </defs>

      <!-- Apple body shape -->
      <path d="M 100,40
               C 70,40 45,65 45,105
               C 45,145 70,175 100,175
               C 130,175 155,145 155,105
               C 155,65 130,40 100,40 Z"
            fill="url(#appleGradient)" />

      <!-- Spots overlay -->
      ${spots}

      <!-- Highlight shine -->
      <ellipse cx="80" cy="75" rx="20" ry="28" fill="url(#highlight)" />

      <!-- Stem -->
      <path d="M 100,45 Q 105,30 110,25 L 108,22 Q 102,28 98,42 Z"
            fill="#4a3520" />

      <!-- Leaf -->
      <path d="M 110,28 Q 125,22 130,32 Q 122,38 110,32 Z"
            fill="#3d6b2c" />
    </svg>
  `;
}