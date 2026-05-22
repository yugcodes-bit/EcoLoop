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

    appearance: {
      bodyColor:   "#d9342b",
      highlightColor: "#f56358",
      shadowColor: "#a01b14",
      hasSpots:    false,
      spotIntensity: 0
    },

    truth: {
      class: "Fresh",
      classCode: 0,
      pH: 4.25,
      internalDescription: "Healthy white-cream flesh, intact cells, no rot"
    },

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
   SVG Apple Generator — Whole apple (outside view)
   ============================================================ */

function getAppleSVG(appearance, size) {
  size = size || 200;
  const { bodyColor, highlightColor, shadowColor, hasSpots, spotIntensity } = appearance;

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

      <path d="M 100,40
               C 70,40 45,65 45,105
               C 45,145 70,175 100,175
               C 130,175 155,145 155,105
               C 155,65 130,40 100,40 Z"
            fill="url(#appleGradient)" />

      ${spots}

      <ellipse cx="80" cy="75" rx="20" ry="28" fill="url(#highlight)" />

      <path d="M 100,45 Q 105,30 110,25 L 108,22 Q 102,28 98,42 Z"
            fill="#4a3520" />

      <path d="M 110,28 Q 125,22 130,32 Q 122,38 110,32 Z"
            fill="#3d6b2c" />
    </svg>
  `;
}


/* ============================================================
   Apple Cross-Section SVG — Reveals internal state when cut
   ============================================================ */

function getAppleCrossSectionSVG(truth, side, size) {
  side = side || "left";
  size = size || 200;

  const isLeft = side === "left";
  const classCode = truth.classCode;

  let rotPatches = '';

  if (classCode === 0) {
    rotPatches = '';
  } else if (classCode === 1) {
    rotPatches = `
      <ellipse cx="${isLeft ? 130 : 70}" cy="100" rx="18" ry="14"
               fill="#9a6840" opacity="0.5" />
      <ellipse cx="${isLeft ? 125 : 75}" cy="115" rx="10" ry="8"
               fill="#7a4828" opacity="0.4" />
    `;
  } else {
    rotPatches = `
      <ellipse cx="${isLeft ? 130 : 70}" cy="100" rx="32" ry="26"
               fill="#5a3018" opacity="0.85" />
      <ellipse cx="${isLeft ? 115 : 85}" cy="80" rx="18" ry="14"
               fill="#6e3a1f" opacity="0.7" />
      <ellipse cx="${isLeft ? 135 : 65}" cy="140" rx="20" ry="16"
               fill="#4a2810" opacity="0.75" />
      <ellipse cx="${isLeft ? 110 : 90}" cy="120" rx="12" ry="10"
               fill="#3a1f0a" opacity="0.6" />
    `;
  }

  const halfPath = isLeft
    ? "M 140,40 C 110,40 85,65 85,105 C 85,145 110,175 140,175 Z"
    : "M 60,40 C 90,40 115,65 115,105 C 115,145 90,175 60,175 Z";

  const skinColor = "#a52a1f";
  const coreX = isLeft ? 140 : 60;

  return `
    <svg width="${size}" height="${size}" viewBox="0 0 200 200"
         xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="fleshGrad-${side}" cx="50%" cy="40%" r="60%">
          <stop offset="0%"   stop-color="#fbf6e4" />
          <stop offset="70%"  stop-color="#f0e6cc" />
          <stop offset="100%" stop-color="#d9c89c" />
        </radialGradient>
      </defs>

      <path d="${halfPath}" fill="url(#fleshGrad-${side})" />
      <path d="${halfPath}" fill="none" stroke="${skinColor}" stroke-width="3" />

      ${rotPatches}

      <line x1="${coreX}" y1="50" x2="${coreX}" y2="165"
            stroke="#c4a878" stroke-width="1" opacity="0.5" />

      <ellipse cx="${coreX + (isLeft ? -8 : 8)}" cy="100"
               rx="3" ry="5" fill="#5a3a20" opacity="0.7" />
      <ellipse cx="${coreX + (isLeft ? -8 : 8)}" cy="115"
               rx="3" ry="5" fill="#5a3a20" opacity="0.7" />
    </svg>
  `;
}