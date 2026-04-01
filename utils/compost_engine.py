# ============================================================
# ECO-LOOP — Compost Valorization Engine
# utils/compost_engine.py
#
# Takes predicted rot class and pH value as input
# Returns a complete compost amendment prescription
# No sensor or ML model needed — pure decision logic
# ============================================================

from datetime import datetime

# ── 6-Tier pH Decision Table ───────────────────────────────────────────────
# Based on composting science literature (Torres et al. 2016)
# Each tier maps a pH range to a specific amendment prescription

COMPOST_TIERS = [
    {
        "ph_min": 0.0,
        "ph_max": 3.0,
        "rot_stage": "Severe Rot",
        "cn_ratio": "30:1",
        "primary_amendment": "Crushed eggshells + wood ash + dry leaves",
        "secondary_amendment": "Heavy addition of dry carbon-rich matter",
        "action": "Route to composting — apply strong alkaline amendment immediately",
        "expected_compost_ready": "8-10 weeks with amendment",
        "warning": "Highly acidic — do not compost without amendment"
    },
    {
        "ph_min": 3.0,
        "ph_max": 3.5,
        "rot_stage": "High Rot",
        "cn_ratio": "25:1",
        "primary_amendment": "Dry straw + cardboard shreds",
        "secondary_amendment": "Small amount of lime powder",
        "action": "Route to composting — apply amendment before adding to pile",
        "expected_compost_ready": "6-8 weeks with amendment",
        "warning": "Acidic — amendment required for efficient composting"
    },
    {
        "ph_min": 3.5,
        "ph_max": 4.0,
        "rot_stage": "Early-Moderate Rot",
        "cn_ratio": "20:1",
        "primary_amendment": "60% brown matter (straw/cardboard)",
        "secondary_amendment": "40% green matter (grass/vegetable scraps)",
        "action": "Route to composting — standard balanced mix",
        "expected_compost_ready": "5-7 weeks with amendment",
        "warning": "Mildly acidic — standard amendment sufficient"
    },
    {
        "ph_min": 4.0,
        "ph_max": 5.0,
        "rot_stage": "Mild Rot",
        "cn_ratio": "15:1",
        "primary_amendment": "50% brown matter + 50% green matter",
        "secondary_amendment": "No additional amendment needed",
        "action": "Route to composting — optimal balance zone",
        "expected_compost_ready": "4-6 weeks",
        "warning": "None — suitable for direct composting"
    },
    {
        "ph_min": 5.0,
        "ph_max": 6.0,
        "rot_stage": "Near-Neutral",
        "cn_ratio": "10:1",
        "primary_amendment": "Grass clippings + vegetable scraps",
        "secondary_amendment": "Nitrogen-rich additions recommended",
        "action": "Route to composting — nitrogen-rich mix preferred",
        "expected_compost_ready": "3-5 weeks",
        "warning": "None — excellent composting candidate"
    },
]

# ── Fresh Apple Handler ────────────────────────────────────────────────────
FRESH_RESPONSE = {
    "rot_stage": "Fresh",
    "cn_ratio": "N/A",
    "primary_amendment": "N/A",
    "secondary_amendment": "N/A",
    "action": "Route to human consumption or cold storage",
    "expected_compost_ready": "N/A",
    "warning": "Do NOT compost — this apple is fresh and commercially viable"
}

# ── Class name mapping ─────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Fresh",
    1: "Early-Rot",
    2: "Severe-Rot"
}

# ── Confidence mapping based on pH distance from class boundary ────────────
def get_confidence(predicted_ph, rot_class):
    """
    Returns confidence level based on how far the pH is
    from the nearest class boundary.
    Far from boundary = High confidence
    Close to boundary = Medium or Low confidence
    """
    boundaries = [3.0, 3.5, 4.0, 5.0]
    
    min_distance = min(abs(predicted_ph - b) for b in boundaries)
    
    if min_distance > 0.3:
        return "High"
    elif min_distance > 0.15:
        return "Medium"
    else:
        return "Low — pH is near a class boundary"

# ══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION — this is what the FastAPI server will call
# ══════════════════════════════════════════════════════════════════════════

def get_compost_prescription(rot_class: int, predicted_ph: float) -> dict:
    """
    Main compost engine function.
    
    Parameters:
        rot_class    : int   — 0 (Fresh), 1 (Early-Rot), 2 (Severe-Rot)
        predicted_ph : float — predicted internal pH from regression model
    
    Returns:
        dict — complete prescription with all fields
    """

    # ── Handle Fresh apples first ──────────────────────────────────────────
    if rot_class == 0:
        return {
            "scan_id": f"ECL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "rot_class": 0,
            "rot_class_name": "Fresh",
            "predicted_pH": round(predicted_ph, 2),
            "confidence": get_confidence(predicted_ph, rot_class),
            "compost_recommendation": FRESH_RESPONSE,
            "model_used": "Random Forest",
            "sensor": "AS7265x 18-channel Vis-NIR"
        }

    # ── Find correct pH tier for rotten apples ────────────────────────────
    matched_tier = None

    for tier in COMPOST_TIERS:
        if tier["ph_min"] <= predicted_ph < tier["ph_max"]:
            matched_tier = tier
            break

    # ── Handle edge case: pH outside expected range ────────────────────────
    if matched_tier is None:
        if predicted_ph < COMPOST_TIERS[0]["ph_min"]:
            matched_tier = COMPOST_TIERS[0]   # use most acidic tier
        else:
            matched_tier = COMPOST_TIERS[-1]  # use least acidic tier

    # ── Build complete response ────────────────────────────────────────────
    response = {
        "scan_id": f"ECL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "rot_class": rot_class,
        "rot_class_name": CLASS_NAMES[rot_class],
        "predicted_pH": round(predicted_ph, 2),
        "confidence": get_confidence(predicted_ph, rot_class),
        "compost_recommendation": {
            "rot_stage": matched_tier["rot_stage"],
            "cn_ratio": matched_tier["cn_ratio"],
            "primary_amendment": matched_tier["primary_amendment"],
            "secondary_amendment": matched_tier["secondary_amendment"],
            "action": matched_tier["action"],
            "expected_compost_ready": matched_tier["expected_compost_ready"],
            "warning": matched_tier["warning"]
        },
        "model_used": "Random Forest",
        "sensor": "AS7265x 18-channel Vis-NIR"
    }

    return response


# ══════════════════════════════════════════════════════════════════════════
# TEST — run this file directly to verify the engine works
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("ECO-LOOP Compost Engine — Test Run")
    print("=" * 60)

    # Test all 3 classes with different pH values
    test_cases = [
        {"rot_class": 0, "ph": 4.20, "label": "Fresh apple"},
        {"rot_class": 1, "ph": 3.55, "label": "Early-Rot apple"},
        {"rot_class": 1, "ph": 3.22, "label": "Early-Rot (near boundary)"},
        {"rot_class": 2, "ph": 2.95, "label": "Severe-Rot apple"},
        {"rot_class": 2, "ph": 3.40, "label": "Severe-Rot (higher pH)"},
    ]

    for test in test_cases:
        print(f"\n--- Testing: {test['label']} ---")
        result = get_compost_prescription(test["rot_class"], test["ph"])
        print(f"  Rot Class    : {result['rot_class_name']}")
        print(f"  Predicted pH : {result['predicted_pH']}")
        print(f"  Confidence   : {result['confidence']}")
        print(f"  Rot Stage    : {result['compost_recommendation']['rot_stage']}")
        print(f"  C:N Ratio    : {result['compost_recommendation']['cn_ratio']}")
        print(f"  Amendment    : {result['compost_recommendation']['primary_amendment']}")
        print(f"  Action       : {result['compost_recommendation']['action']}")
        print(f"  Ready In     : {result['compost_recommendation']['expected_compost_ready']}")
        print(f"  Warning      : {result['compost_recommendation']['warning']}")