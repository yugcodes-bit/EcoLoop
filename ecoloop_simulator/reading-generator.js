/* ============================================================
   ECO-LOOP Simulator — Spectral Reading Generator
   Generates realistic 18-channel readings based on apple state
   Anchored to real IISER pilot data
   ============================================================ */

const CHANNEL_WAVELENGTHS = [
  410, 435, 460, 485, 510, 535,
  560, 585, 610, 645, 680, 705,
  730, 760, 810, 860, 900, 940
];

// Key diagnostic bands (highlighted in UI)
const KEY_BAND_INFO = {
  680: "Chlorophyll Band",
  810: "Cell Structure Band",
  940: "Water Band"
};

// Spectral profile anchors — values 0.0 to 1.0
// These match the patterns from your IISER pilot data
const SPECTRAL_PROFILES = {
  fresh: {
    // High reflectance overall, deep dip at 680nm, strong peak at 810nm
    base: [0.04, 0.05, 0.05, 0.05, 0.06, 0.07,
           0.10, 0.13, 0.18, 0.28, 0.36, 0.43,
           0.46, 0.49, 0.50, 0.50, 0.49, 0.46],
    variance: 0.025
  },
  early: {
    // Reduced reflectance, less pronounced dip at 680, lower peak at 810
    base: [0.04, 0.05, 0.06, 0.07, 0.09, 0.11,
           0.14, 0.16, 0.19, 0.24, 0.27, 0.31,
           0.34, 0.37, 0.39, 0.39, 0.37, 0.34],
    variance: 0.030
  },
  severe: {
    // Low reflectance, almost no dip at 680, very low peak at 810
    base: [0.03, 0.04, 0.04, 0.05, 0.06, 0.08,
           0.10, 0.12, 0.14, 0.17, 0.20, 0.23,
           0.26, 0.29, 0.30, 0.30, 0.29, 0.27],
    variance: 0.025
  }
};

/* ────────────────────────────────────────────────────────
   Main function — generate 18 readings for an apple
   ──────────────────────────────────────────────────────── */
function generateSpectralReading(spectralProfile) {
  const profile = SPECTRAL_PROFILES[spectralProfile] || SPECTRAL_PROFILES.fresh;

  return CHANNEL_WAVELENGTHS.map((wavelength, i) => {
    const baseValue = profile.base[i];
    const noise = (Math.random() - 0.5) * 2 * profile.variance;
    const value = Math.max(0.0, Math.min(1.0, baseValue + noise));

    return {
      wavelength,
      value: Number(value.toFixed(4)),
      isKeyBand: wavelength in KEY_BAND_INFO,
      keyBandInfo: KEY_BAND_INFO[wavelength] || null
    };
  });
}   