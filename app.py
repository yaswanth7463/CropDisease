"""
Crop Disease Detector — Anantapur Edition
Weather-aware remedy advisor for Anantapur, Andhra Pradesh.

Setup:
  1. Get a FREE OpenWeatherMap API key at https://openweathermap.org/api
  2. On HuggingFace Spaces → Settings → Variables and Secrets
     Add: WEATHER_API_KEY = <your key>
  3. Deploy — weather will auto-update on every prediction.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import os
from datetime import datetime

# ─── Anantapur GPS Coordinates ───────────────────────────────────────────────
LAT = 14.6819
LON = 77.5999
CITY = "Anantapur, Andhra Pradesh"

# ─── Supported Crops ─────────────────────────────────────────────────────────
SUPPORTED_CROPS = [
    "Apple", "Blueberry", "Cherry", "Corn (Maize)", "Grape", "Orange",
    "Peach", "Bell Pepper", "Potato", "Raspberry", "Soybean", "Squash",
    "Strawberry", "Tomato",
    "Groundnut", "Mango", "Banana", "Chilli", "Lemon",
    "Cotton", "Sunflower", "Paddy (Rice)", "Pomegranate"
]

# ─── Local Availability Tags ─────────────────────────────────────────────────
# How easy it is to find each chemical in Anantapur's agri shops
LOCAL = {
    "mancozeb":          "✅ Widely available in Anantapur",
    "copper oxychloride":"✅ Widely available in Anantapur",
    "carbendazim":       "✅ Widely available in Anantapur",
    "chlorothalonil":    "✅ Widely available in Anantapur",
    "imidacloprid":      "✅ Widely available in Anantapur",
    "neem oil":          "✅ Widely available — very affordable locally",
    "copper hydroxide":  "✅ Widely available in Anantapur",
    "streptocycline":    "🟡 Available at district agri office / KVK",
    "propiconazole":     "🟡 Available at larger agri shops in Anantapur town",
    "hexaconazole":      "🟡 Available at larger agri shops",
    "metalaxyl":         "🟡 Available at KVK or Kurnool",
    "trichoderma":       "🟡 Available at KVK Reddipalli — sometimes subsidised",
    "azoxystrobin":      "🔴 May need to order from Kurnool or Bengaluru",
    "spinosad":          "🔴 May need to order from Kurnool or Bengaluru",
    "tricyclazole":      "🟡 Available at larger agri shops in Anantapur town",
    "iprodione":         "🟡 Available at larger agri shops",
    "triadimefon":       "🟡 Available at larger agri shops",
    "wettable sulfur":   "✅ Widely available in Anantapur",
}

# ─── Disease Remedies ─────────────────────────────────────────────────────────
# Each remedy has:
#   description: what the disease is
#   severity:    None / Low / Moderate / High / Critical
#   remedy:      list of treatment steps (plain language)
#   spray_ok_in_rain: False means this remedy needs dry weather to work
#   heat_note:   extra advice for Anantapur's hot dry climate
#   key_chemical: main chemical for local availability lookup

disease_remedies = {

    # ═════════════════════════════════════
    # PLANTVILLAGE ORIGINAL (14 crops)
    # ═════════════════════════════════════

    "Apple___Apple_scab": {
        "description": "Fungal disease causing olive-green to black spots on leaves and fruit.",
        "severity": "Moderate",
        "remedy": ["Apply captan or myclobutanil fungicide at bud break", "Rake and destroy fallen leaves", "Prune trees to improve air circulation"],
        "spray_ok_in_rain": False, "heat_note": "Spray early morning before 8 AM in summer.", "key_chemical": "mancozeb"
    },
    "Apple___Black_rot": {
        "description": "Fungal disease causing brown leaf spots and rotting fruit with black lesions.",
        "severity": "High",
        "remedy": ["Remove and destroy mummified fruits and dead wood", "Apply copper-based fungicide", "Prune infected branches 15 inches below visible infection"],
        "spray_ok_in_rain": False, "heat_note": "Avoid spraying in midday heat above 38°C.", "key_chemical": "copper oxychloride"
    },
    "Apple___Cedar_apple_rust": {
        "description": "Fungal disease causing bright orange spots on leaves and fruit.",
        "severity": "Moderate",
        "remedy": ["Apply myclobutanil at pink bud stage", "Remove nearby cedar trees if possible", "Spray every 7-10 days during wet weather"],
        "spray_ok_in_rain": False, "heat_note": "Spray in cool morning hours.", "key_chemical": "mancozeb"
    },
    "Apple___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Continue regular care and monitoring"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Blueberry___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Maintain soil pH 4.5-5.5", "Water regularly"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Fungal disease causing white powdery coating on leaves and shoots.",
        "severity": "Moderate",
        "remedy": ["Apply wettable sulfur or potassium bicarbonate fungicide", "Prune affected shoots", "Water at the base only"],
        "spray_ok_in_rain": False, "heat_note": "Sulfur burns leaves above 35°C — use only in early morning.", "key_chemical": "wettable sulfur"
    },
    "Cherry_(including_sour)___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Continue regular care"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Fungal disease causing rectangular gray to tan lesions on leaves.",
        "severity": "High",
        "remedy": ["Apply strobilurin or triazole fungicide early", "Plant resistant hybrids", "Rotate crops annually", "Till infected residue after harvest"],
        "spray_ok_in_rain": False, "heat_note": "High humidity in Anantapur's kharif season increases infection — act immediately if spotted.", "key_chemical": "mancozeb"
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Fungal disease causing orange-brown pustules on leaves.",
        "severity": "Low",
        "remedy": ["Apply fungicide at early stages", "Plant rust-resistant varieties next season"],
        "spray_ok_in_rain": False, "heat_note": "In Anantapur's dry climate, rust is less severe — monitor but may not need spraying.", "key_chemical": "mancozeb"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Fungal disease causing long cigar-shaped gray-green lesions on leaves.",
        "severity": "High",
        "remedy": ["Apply propiconazole at early tasseling", "Plant resistant hybrids", "Rotate with soybeans or groundnut", "Remove crop residue after harvest"],
        "spray_ok_in_rain": False, "heat_note": "Common during Anantapur's kharif — spray before tassel emergence.", "key_chemical": "propiconazole"
    },
    "Corn_(maize)___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Maintain nutrients and irrigation"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Grape___Black_rot": {
        "description": "Fungal disease causing brown leaf spots and shriveled black mummified berries.",
        "severity": "High",
        "remedy": ["Apply mancozeb fungicide before bloom", "Remove mummified berries and infected shoots", "Prune for air circulation and spray every 10-14 days"],
        "spray_ok_in_rain": False, "heat_note": "Spray in cool morning hours. Mancozeb is widely available in Anantapur.", "key_chemical": "mancozeb"
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Complex fungal disease causing tiger-stripe patterns on leaves and berry spotting.",
        "severity": "High",
        "remedy": ["Prune infected wood and seal wounds", "Remove severely infected vines", "Avoid pruning during wet weather"],
        "spray_ok_in_rain": True, "heat_note": "Prune during Anantapur's dry season (Dec-Feb) to avoid wound infection.", "key_chemical": "copper oxychloride"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Fungal disease causing angular dark brown lesions on leaves.",
        "severity": "Moderate",
        "remedy": ["Apply copper-based fungicide", "Improve air circulation", "Avoid overhead irrigation"],
        "spray_ok_in_rain": False, "heat_note": "Use drip irrigation — very suitable for Anantapur's scarce water conditions.", "key_chemical": "copper oxychloride"
    },
    "Grape___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Maintain pruning and monitoring"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Bacterial disease causing yellowing and misshapen fruit. No cure exists.",
        "severity": "Critical",
        "remedy": ["Remove and destroy infected trees immediately", "Control Asian citrus psyllid with systemic insecticide", "Use certified disease-free planting material", "Report to agricultural authorities"],
        "spray_ok_in_rain": True, "heat_note": "Contact Anantapur district horticulture officer immediately.", "key_chemical": "imidacloprid"
    },
    "Peach___Bacterial_spot": {
        "description": "Bacterial disease causing water-soaked spots on leaves and twigs.",
        "severity": "Moderate",
        "remedy": ["Apply copper-based bactericide during early leaf development", "Plant resistant varieties", "Avoid overhead irrigation"],
        "spray_ok_in_rain": False, "heat_note": "Use drip irrigation in Anantapur — saves water and prevents this disease.", "key_chemical": "copper oxychloride"
    },
    "Peach___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Monitor for borers and aphids"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial disease causing scab-like spots on leaves and fruit.",
        "severity": "Moderate",
        "remedy": ["Apply copper bactericide weekly during wet weather", "Use certified disease-free seeds", "Rotate crops 2-3 years"],
        "spray_ok_in_rain": False, "heat_note": "Copper oxychloride is easily available in Anantapur at low cost.", "key_chemical": "copper oxychloride"
    },
    "Pepper,_bell___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Maintain watering and monitor for aphids"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Potato___Early_blight": {
        "description": "Fungal disease causing dark brown spots with concentric rings on leaves.",
        "severity": "Moderate",
        "remedy": ["Apply chlorothalonil or mancozeb every 7-10 days", "Remove infected lower leaves", "Rotate crops and use certified seed potatoes"],
        "spray_ok_in_rain": False, "heat_note": "Spray early morning — both chemicals are widely available in Anantapur.", "key_chemical": "mancozeb"
    },
    "Potato___Late_blight": {
        "description": "Serious disease causing dark water-soaked lesions with white mold.",
        "severity": "High",
        "remedy": ["Apply mancozeb or metalaxyl fungicide immediately", "Remove and destroy infected plants", "Harvest tubers soon if infection is severe"],
        "spray_ok_in_rain": False, "heat_note": "Rare in Anantapur's dry climate — but watch during northeast monsoon (Oct-Nov).", "key_chemical": "mancozeb"
    },
    "Potato___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Maintain watering and hill soil around plants"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Raspberry___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Prune old canes after fruiting"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Soybean___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Monitor for aphids and spider mites"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Squash___Powdery_mildew": {
        "description": "White powdery fungal coating on leaves.",
        "severity": "Moderate",
        "remedy": ["Apply neem oil (5ml/L) or wettable sulfur", "Remove infected leaves", "Improve air circulation by wider plant spacing"],
        "spray_ok_in_rain": False, "heat_note": "Neem oil is cheapest and most available option in Anantapur. Do not spray sulfur above 35°C.", "key_chemical": "neem oil"
    },
    "Strawberry___Leaf_scorch": {
        "description": "Fungal disease causing purple to brown spots on leaves.",
        "severity": "Moderate",
        "remedy": ["Apply captan fungicide", "Remove infected leaves", "Renovate beds annually"],
        "spray_ok_in_rain": False, "heat_note": "Spray early morning in Anantapur's hot months.", "key_chemical": "mancozeb"
    },
    "Strawberry___healthy": {"description": "Healthy plant.", "severity": "None", "remedy": ["Maintain watering and mulching"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},
    "Tomato___Bacterial_spot": {
        "description": "Bacterial disease causing small water-soaked spots on leaves and fruit.",
        "severity": "Moderate",
        "remedy": ["Apply copper bactericide (3g/L) every 7-10 days", "Use disease-free seeds", "Avoid working in field when plants are wet", "Rotate crops every 2-3 years"],
        "spray_ok_in_rain": False, "heat_note": "Copper oxychloride is cheap and widely available at any agri shop in Anantapur.", "key_chemical": "copper oxychloride"
    },
    "Tomato___Early_blight": {
        "description": "Fungal disease causing dark brown spots with yellow rings on leaves.",
        "severity": "Moderate",
        "remedy": ["Remove and destroy infected leaves immediately", "Apply mancozeb (2.5g/L) every 7-10 days", "Water at the base — avoid wetting foliage"],
        "spray_ok_in_rain": False, "heat_note": "Mancozeb is the most affordable and available option in Anantapur.", "key_chemical": "mancozeb"
    },
    "Tomato___Late_blight": {
        "description": "Serious disease causing large dark lesions with white mold.",
        "severity": "High",
        "remedy": ["Apply chlorothalonil or mancozeb immediately", "Remove and bag infected material — do not compost", "Destroy all debris at end of season"],
        "spray_ok_in_rain": False, "heat_note": "Watch for this during northeast monsoon. Act within 24 hours of spotting.", "key_chemical": "mancozeb"
    },
    "Tomato___Leaf_Mold": {
        "description": "Yellow patches on upper leaf with olive-green mold below.",
        "severity": "Moderate",
        "remedy": ["Improve ventilation in greenhouse", "Apply copper-based fungicide", "Remove infected leaves promptly"],
        "spray_ok_in_rain": False, "heat_note": "Mainly in greenhouse tomatoes. Open-field tomatoes in Anantapur less affected.", "key_chemical": "copper oxychloride"
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Small circular spots with dark borders and gray centers on leaves.",
        "severity": "Moderate",
        "remedy": ["Apply mancozeb or chlorothalonil fungicide", "Remove infected lower leaves", "Mulch around plants to prevent soil splash"],
        "spray_ok_in_rain": False, "heat_note": "Occurs during kharif tomato season. Spray early morning.", "key_chemical": "mancozeb"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Pest causing stippled yellow leaves with fine webbing on undersides.",
        "severity": "Moderate",
        "remedy": ["Apply neem oil (5ml/L) weekly", "Spray water forcefully on leaf undersides", "Introduce predatory mites as biological control"],
        "spray_ok_in_rain": False, "heat_note": "Very common in Anantapur's hot dry weather (March-May). Neem oil is cheapest solution.", "key_chemical": "neem oil"
    },
    "Tomato___Target_Spot": {
        "description": "Brown lesions with concentric rings on leaves and fruit.",
        "severity": "Moderate",
        "remedy": ["Apply azoxystrobin or mancozeb fungicide", "Remove infected material", "Improve air circulation through pruning"],
        "spray_ok_in_rain": False, "heat_note": "Mancozeb is a cheaper locally available alternative to azoxystrobin.", "key_chemical": "mancozeb"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease from whiteflies — yellowing, curling leaves and severely stunted plants.",
        "severity": "High",
        "remedy": ["Control whiteflies with imidacloprid (0.3ml/L) immediately", "Remove infected plants immediately", "Use reflective silver mulch to repel whiteflies", "Plant TYLCV-resistant varieties next season"],
        "spray_ok_in_rain": False, "heat_note": "Imidacloprid is widely available in Anantapur. Whitefly pressure is highest in hot dry months.", "key_chemical": "imidacloprid"
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Viral disease causing mosaic yellow-green mottling and distorted leaves.",
        "severity": "High",
        "remedy": ["Remove and destroy infected plants — no cure exists", "Disinfect tools with bleach solution", "Plant virus-resistant varieties"],
        "spray_ok_in_rain": True, "heat_note": "Spreads through contact — wash hands before handling plants.", "key_chemical": ""
    },
    "Tomato___healthy": {"description": "Healthy tomato plant.", "severity": "None", "remedy": ["Continue regular watering and monitoring"], "spray_ok_in_rain": True, "heat_note": "", "key_chemical": ""},

    # ═════════════════════════════════════════════
    # ANANTAPUR SPECIFIC CROPS
    # ═════════════════════════════════════════════

    # 🥜 GROUNDNUT — dominant crop of Anantapur (86% of area)
    "Groundnut___Early_leaf_spot": {
        "description": "Fungal disease causing circular brown spots with yellow halos — one of the most damaging groundnut diseases in Anantapur.",
        "severity": "High",
        "remedy": [
            "Spray chlorothalonil (2g/L) or mancozeb (2.5g/L) every 10-14 days",
            "Start spraying 30-35 days after sowing — do NOT wait for severe symptoms",
            "Remove and destroy infected plant debris after harvest",
            "Plant resistant varieties TAG-24 or K-6",
            "Maintain 30cm row spacing for air circulation in Anantapur's humid kharif"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Most critical disease for Anantapur. Spray in morning (before 9 AM) to avoid afternoon heat degrading the chemical. Both mancozeb and chlorothalonil are widely available and affordable here.",
        "key_chemical": "mancozeb"
    },
    "Groundnut___Late_leaf_spot": {
        "description": "More severe than early leaf spot — dark brown to black circular spots leading to complete defoliation.",
        "severity": "High",
        "remedy": [
            "Apply mancozeb (2g/L) + carbendazim (1g/L) tank mix at first appearance",
            "Spray every 10-12 days during humid monsoon weather",
            "Improve field drainage — Anantapur's red soil drains well, but low-lying patches do not",
            "Plant resistant varieties ICGV 86031 or VRI 2",
            "Rotate with jowar or maize the next season"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Hits hardest during Anantapur's kharif (July-Sep) when humidity is highest. Both chemicals available cheaply at any agri shop in Anantapur.",
        "key_chemical": "mancozeb"
    },
    "Groundnut___Rust": {
        "description": "Fungal disease causing orange-brown pustules on leaf undersides — drastically reduces pod fill.",
        "severity": "High",
        "remedy": [
            "Spray propiconazole (1ml/L) or triadimefon (1g/L) at first rust appearance",
            "Repeat every 14 days — do not miss the second spray",
            "Avoid planting volunteer groundnut plants nearby — they harbour rust",
            "Plant rust-resistant varieties (ICGV 86590, VRI 2)",
            "Scout fields from 45 days after sowing, especially after cloudy days"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Rust is common in Anantapur during high-humidity spells in August-September. Propiconazole is available at larger agri shops in Anantapur town.",
        "key_chemical": "propiconazole"
    },
    "Groundnut___Alternaria_blight": {
        "description": "Fungal disease causing irregular brown spots with concentric rings on older leaves.",
        "severity": "Moderate",
        "remedy": [
            "Apply mancozeb (2.5g/L) at first appearance",
            "Remove infected leaves promptly",
            "Ensure proper field drainage in low-lying areas",
            "Avoid excess nitrogen — promotes soft growth susceptible to disease"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Less severe than leaf spot and rust in Anantapur. Mancozeb handles all three — use the same spray programme.",
        "key_chemical": "mancozeb"
    },
    "Groundnut___healthy": {
        "description": "The groundnut plant appears healthy — great news for Anantapur's most important crop!",
        "severity": "None",
        "remedy": [
            "Apply gypsum (200 kg/ha) at pod formation stage — essential for Anantapur's red soils",
            "Ensure adequate boron nutrition for pod development",
            "Scout for leaf spot from 30 days after sowing as a preventive habit"
        ],
        "spray_ok_in_rain": True, "heat_note": "Gypsum application at pegging stage is critical in Anantapur's red soils for calcium nutrition.", "key_chemical": ""
    },

    # 🥭 MANGO
    "Mango___Anthracnose": {
        "description": "Most common mango disease in Anantapur — dark sunken lesions on leaves, flowers, and fruit. Major cause of post-harvest losses.",
        "severity": "High",
        "remedy": [
            "Spray copper oxychloride (3g/L) before flowering — do not miss this",
            "Repeat at 15-day intervals during entire flowering period",
            "Remove and destroy infected flowers and fruit",
            "Use drip irrigation — avoids wetting foliage which spreads disease",
            "Post-harvest: hot water treatment (52°C for 5 min) before storage"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Copper oxychloride is cheap and widely available in Anantapur. Critical spray timing is panicle emergence — do not delay.",
        "key_chemical": "copper oxychloride"
    },
    "Mango___Powdery_mildew": {
        "description": "White powdery growth on flowers and young leaves — causes massive flower drop and total yield loss.",
        "severity": "High",
        "remedy": [
            "Spray wettable sulfur (2g/L) at panicle emergence — this is the critical moment",
            "Repeat every 10 days through full flowering",
            "Avoid excessive nitrogen fertilizer in October-November",
            "Ensure tree spacing of at least 8m for air circulation"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Very common during Anantapur mango season (Jan-Mar). Wettable sulfur is widely available. Do NOT spray sulfur if temperature above 35°C — use triadimefon instead.",
        "key_chemical": "wettable sulfur"
    },
    "Mango___Die_back": {
        "description": "Fungal disease causing tip-to-base drying of twigs — leaves turn brown and stick to the branch.",
        "severity": "High",
        "remedy": [
            "Prune and burn infected twigs 4-6 inches below visible infection",
            "Apply copper oxychloride paste on cut surfaces immediately",
            "Spray carbendazim (1g/L) on whole tree after pruning",
            "Prune only during Anantapur's dry season (Dec-Feb) — not during monsoon"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Both copper oxychloride and carbendazim are easily available in Anantapur. Dry-season pruning is the key preventive habit.",
        "key_chemical": "copper oxychloride"
    },
    "Mango___Bacterial_canker": {
        "description": "Bacterial disease causing water-soaked lesions with yellow halos on leaves, stems, and fruit.",
        "severity": "High",
        "remedy": [
            "Spray streptocycline (200ppm) + copper oxychloride (3g/L) immediately",
            "Remove infected plant parts and burn them — do not compost",
            "Avoid pruning during northeast monsoon rains (Oct-Nov)",
            "Disinfect pruning tools with 10% bleach solution between each tree"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Streptocycline is available at district agri office and larger shops in Anantapur town. Combine with copper oxychloride which is widely available.",
        "key_chemical": "streptocycline"
    },
    "Mango___Sooty_mold": {
        "description": "Black coating on leaves from insect honeydew — reduces photosynthesis and makes fruit unmarketable.",
        "severity": "Moderate",
        "remedy": [
            "Spray imidacloprid (0.3ml/L) to kill sap-sucking insects (mealybugs, aphids) first",
            "Wash leaves with diluted soap solution (2g/L) to physically remove mold",
            "Apply neem oil (5ml/L) as ongoing eco-friendly insecticide"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Imidacloprid and neem oil are both widely and cheaply available across Anantapur district.",
        "key_chemical": "imidacloprid"
    },
    "Mango___healthy": {
        "description": "The mango tree appears healthy.",
        "severity": "None",
        "remedy": ["Apply zinc sulfate (5g/L) spray after harvest", "Monitor for mango hopper at flowering", "Apply prophylactic copper spray before northeast monsoon"],
        "spray_ok_in_rain": True, "heat_note": "Anantapur's dry climate generally suits mango well. Irrigation during fruit development is critical.", "key_chemical": ""
    },

    # 🍌 BANANA
    "Banana___Sigatoka_(Yellow_Sigatoka)": {
        "description": "Fungal disease causing yellow streaks turning brown on leaves — reduces photosynthesis and fruit quality.",
        "severity": "Moderate",
        "remedy": [
            "Spray propiconazole (0.1%) or mancozeb (0.2%) every 3 weeks",
            "Remove badly infected leaves at the petiole — burn them",
            "Ensure good drainage — banana suffers in waterlogged red soil",
            "Apply potassium chloride (MOP) fertilizer to improve resistance"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Mancozeb is widely available in Anantapur. Propiconazole is available at larger shops. Banana does well in Anantapur with drip irrigation.",
        "key_chemical": "mancozeb"
    },
    "Banana___Black_Sigatoka": {
        "description": "More aggressive than Yellow Sigatoka — black streaks destroy 50-80% of leaf area if untreated.",
        "severity": "High",
        "remedy": [
            "Apply propiconazole or tridemorph systemic fungicide immediately",
            "Alternate between fungicide types every 2 sprays to avoid resistance",
            "Remove and destroy all heavily infected leaves",
            "Plant resistant varieties Grand Naine or Robusta"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "More severe during monsoon months in Anantapur. Propiconazole available at agri shops in town.",
        "key_chemical": "propiconazole"
    },
    "Banana___Panama_wilt_(Fusarium_wilt)": {
        "description": "Devastating soil disease — entire plant yellows from old leaves inward and dies. No cure.",
        "severity": "Critical",
        "remedy": [
            "Remove and destroy infected plants immediately — do NOT compost",
            "Do not plant banana in same soil for 20+ years",
            "Plant Cavendish group varieties (resistant to Race 1)",
            "Apply Trichoderma viride biocontrol before planting in new land",
            "Contact Anantapur district horticulture officer for guidance"
        ],
        "spray_ok_in_rain": True,
        "heat_note": "Trichoderma viride is available subsidised at KVK Reddipalli (Anantapur). Call 08554-255560.",
        "key_chemical": "trichoderma"
    },
    "Banana___Bunchy_top_virus": {
        "description": "Viral disease from aphids — dark green streaks on leaves, plant stunted with leaves bunched at top. No cure.",
        "severity": "Critical",
        "remedy": [
            "Destroy infected plants immediately — there is no treatment",
            "Control banana aphid vector with imidacloprid (0.3ml/L)",
            "Use only virus-indexed tissue culture planting material",
            "Maintain 10-metre buffer zone around infected plant"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Imidacloprid is widely available in Anantapur. Use tissue culture planting material from certified nurseries only.",
        "key_chemical": "imidacloprid"
    },
    "Banana___healthy": {
        "description": "The banana plant appears healthy.",
        "severity": "None",
        "remedy": ["Remove dry leaves regularly", "Apply potassium fertilizer for bunch development", "Use drip irrigation — saves water in Anantapur's dry climate"],
        "spray_ok_in_rain": True, "heat_note": "Drip irrigation is ideal for banana in Anantapur — conserves scarce water resources.", "key_chemical": ""
    },

    # 🌶️ CHILLI
    "Chilli___Anthracnose": {
        "description": "Fungal disease causing dark sunken lesions on fruit — major problem in Andhra Pradesh chilli farms.",
        "severity": "High",
        "remedy": [
            "Spray carbendazim (0.1%) or mancozeb (0.2%) from fruit development stage",
            "Repeat every 10-12 days throughout fruiting",
            "Harvest fruit at proper maturity — do not leave over-ripe fruit on plant",
            "Use certified disease-free seed from Guntur or Warangal seed companies",
            "Store harvested chilli in cool dry shade — not in direct sun"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Both carbendazim and mancozeb are very cheap and available at every agri shop in Anantapur. Highly recommended combination.",
        "key_chemical": "carbendazim"
    },
    "Chilli___Leaf_curl_virus": {
        "description": "Thrips-spread viral disease causing severe leaf curling and stunted plants — major yield killer.",
        "severity": "High",
        "remedy": [
            "Spray spinosad (0.5ml/L) or imidacloprid (0.3ml/L) to kill thrips immediately",
            "Remove and destroy infected plants early — every day of delay spreads it further",
            "Install silver reflective mulch to repel thrips — highly effective and reusable",
            "Plant tolerant variety Pusa Jwala or LCA 334",
            "Keep field weed-free — weeds harbour thrips and virus"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Imidacloprid is the most affordable option and widely available in Anantapur. Spinosad may need to be sourced from Kurnool. Thrips are worst in Anantapur's hot dry months (Feb-May).",
        "key_chemical": "imidacloprid"
    },
    "Chilli___Powdery_mildew": {
        "description": "White powdery fungal growth on chilli leaves and stems.",
        "severity": "Moderate",
        "remedy": [
            "Spray wettable sulfur (3g/L) or hexaconazole (0.5ml/L)",
            "Remove badly infected plant parts before spraying",
            "Avoid excess nitrogen and waterlogging"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Wettable sulfur is cheapest available option. Do NOT spray sulfur above 35°C — use hexaconazole in hot months.",
        "key_chemical": "wettable sulfur"
    },
    "Chilli___Bacterial_wilt": {
        "description": "Soil-borne bacterial disease — entire plant wilts suddenly, leaves stay green initially.",
        "severity": "High",
        "remedy": [
            "Remove and destroy wilted plants immediately",
            "Drench surrounding soil with copper oxychloride (3g/L) + streptocycline (200ppm)",
            "Improve field drainage — do not allow water to pond in red soil",
            "Rotate with maize, jowar or groundnut for 3-4 years",
            "Apply Trichoderma viride biocontrol before next planting"
        ],
        "spray_ok_in_rain": True,
        "heat_note": "Copper oxychloride is widely available. Trichoderma available at KVK Reddipalli at low cost. Wilt spreads fastest in waterlogged Anantapur soils during kharif.",
        "key_chemical": "copper oxychloride"
    },
    "Chilli___healthy": {
        "description": "The chilli plant appears healthy.",
        "severity": "None",
        "remedy": ["Apply balanced NPK at flowering", "Monitor for thrips from seedling stage", "Use yellow sticky traps in field for early warning"],
        "spray_ok_in_rain": True, "heat_note": "Yellow sticky traps are low cost and available at KVK. Essential for thrips monitoring in Anantapur.", "key_chemical": ""
    },

    # 🍋 LEMON / CITRUS
    "Lemon___Citrus_canker": {
        "description": "Bacterial disease causing raised corky lesions on leaves, stems, and fruit — spreads by wind-driven rain.",
        "severity": "High",
        "remedy": [
            "Spray copper hydroxide (2g/L) preventively before northeast monsoon",
            "Remove and burn all infected plant parts immediately",
            "Disinfect pruning tools with 10% bleach between each tree",
            "Plant only certified canker-free nursery plants",
            "Install windbreaks on northeast-facing borders — wind spreads bacteria"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Copper hydroxide is widely available in Anantapur. Preventive spray before northeast monsoon (Sep) is the single most important action for lemon growers here.",
        "key_chemical": "copper hydroxide"
    },
    "Lemon___Powdery_mildew": {
        "description": "White powdery coating on young lemon leaves and new shoots.",
        "severity": "Moderate",
        "remedy": [
            "Spray wettable sulfur (2g/L) or triadimefon at first appearance",
            "Remove heavily infected shoots",
            "Reduce nitrogen fertilizer — excess promotes soft new growth"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Do not use sulfur above 35°C in Anantapur's summer — use triadimefon instead. Wettable sulfur is cheapest available option.",
        "key_chemical": "wettable sulfur"
    },
    "Lemon___Greening_(Citrus_HLB)": {
        "description": "Devastating bacterial disease — blotchy yellowing and misshapen bitter fruit. No cure. Spreads via psyllid insect.",
        "severity": "Critical",
        "remedy": [
            "Remove infected trees immediately and destroy completely — do not delay",
            "Control Asian citrus psyllid with imidacloprid systemic insecticide",
            "Use only certified disease-free budwood and rootstocks",
            "Report immediately to Anantapur District Horticulture Officer",
            "There is NO cure — every day of delay means more trees get infected"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Imidacloprid is widely available in Anantapur. Report to Horticulture Officer immediately — this is a notifiable disease. Contact: 08554-274400.",
        "key_chemical": "imidacloprid"
    },
    "Lemon___Sooty_mold": {
        "description": "Black fungal coating from insect honeydew — makes fruit unmarketable and reduces photosynthesis.",
        "severity": "Moderate",
        "remedy": [
            "Kill the insect pest (aphids, mealybugs, scale) with imidacloprid first",
            "Wash leaves with soap solution (2g/L) to physically remove the mold",
            "Apply neem oil (5ml/L) as ongoing eco-friendly insect control"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Neem oil and imidacloprid are the cheapest available options in Anantapur and very effective for this problem.",
        "key_chemical": "neem oil"
    },
    "Lemon___healthy": {
        "description": "The lemon plant appears healthy.",
        "severity": "None",
        "remedy": ["Apply zinc sulfate spray (5g/L) in summer", "Scout new flushes for psyllid", "Use drip irrigation — saves water and avoids wetting leaves"],
        "spray_ok_in_rain": True, "heat_note": "Drip irrigation is strongly recommended for lemon in Anantapur's scarce-water conditions.", "key_chemical": ""
    },

    # 🛢️ COTTON
    "Cotton___Bacterial_blight": {
        "description": "Bacterial disease causing angular water-soaked leaf spots — major cotton disease in Anantapur's black cotton soil areas.",
        "severity": "High",
        "remedy": [
            "Spray streptocycline (100ppm) + copper oxychloride (3g/L) immediately",
            "Remove and destroy infected plant debris from field",
            "Use certified acid-delinted seeds treated with carboxin",
            "Plant resistant varieties LH 1556 or MCU 5"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Streptocycline is available at district agri office. Copper oxychloride is cheaply available everywhere. This disease is worst in Anantapur's 12.5% black cotton soil area during monsoon.",
        "key_chemical": "copper oxychloride"
    },
    "Cotton___Alternaria_leaf_spot": {
        "description": "Fungal disease causing brown spots with rings on older cotton leaves.",
        "severity": "Moderate",
        "remedy": [
            "Spray mancozeb (2g/L) at first appearance",
            "Remove infected lower leaves", "Ensure adequate potassium fertilization"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Mancozeb is widely available. Less severe than bacterial blight in Anantapur — treat promptly but not an emergency.",
        "key_chemical": "mancozeb"
    },
    "Cotton___Curl_virus_(CLCuV)": {
        "description": "Whitefly-spread viral disease causing upward leaf curling and dark green veins — major yield loss.",
        "severity": "High",
        "remedy": [
            "Spray imidacloprid (0.3ml/L) or thiamethoxam immediately to control whiteflies",
            "Remove and destroy infected plants in early stage",
            "Plant Bt cotton varieties with CLCuV resistance where certified seeds are available",
            "Install yellow sticky traps to monitor whitefly population"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Imidacloprid is the most affordable and available whitefly control in Anantapur. Whitefly pressure is highest during hot dry spells in kharif.",
        "key_chemical": "imidacloprid"
    },
    "Cotton___healthy": {
        "description": "The cotton plant appears healthy.",
        "severity": "None",
        "remedy": ["Apply square-stage fertilizer", "Scout for bollworm weekly from 30 DAS", "Use pheromone traps for pink bollworm"],
        "spray_ok_in_rain": True, "heat_note": "Pheromone traps are available at KVK Reddipalli — low cost and very effective early warning.", "key_chemical": ""
    },

    # 🌻 SUNFLOWER
    "Sunflower___Alternaria_leaf_blight": {
        "description": "Fungal disease causing brown spots with concentric rings — very common in Anantapur's hot, semi-arid climate.",
        "severity": "Moderate",
        "remedy": [
            "Spray mancozeb (2.5g/L) or iprodione (1.5ml/L) at first appearance",
            "Remove infected lower leaves before spraying",
            "Avoid overhead irrigation — use furrow or drip",
            "Maintain 45cm row spacing for air circulation"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Mancozeb is cheap and widely available in Anantapur — ideal first choice here. Iprodione available at larger agri shops in town.",
        "key_chemical": "mancozeb"
    },
    "Sunflower___Downy_mildew": {
        "description": "Pale yellow patches on upper leaf with white downy growth below — can kill seedlings completely.",
        "severity": "High",
        "remedy": [
            "Treat seeds with metalaxyl (6g/kg) before sowing — most effective prevention",
            "Spray metalaxyl + mancozeb (2g/L) on seedlings at 10-15 days after emergence",
            "Remove and destroy infected seedlings immediately",
            "Plant resistant hybrids KBSH 44 or PAC 36"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Metalaxyl is available at KVK or Kurnool. Seed treatment is the most important and cost-effective step — do not skip it.",
        "key_chemical": "metalaxyl"
    },
    "Sunflower___Charcoal_rot": {
        "description": "Soil-borne disease causing premature drying and grey stem base discoloration — plant falls over.",
        "severity": "High",
        "remedy": [
            "Maintain adequate soil moisture during flowering — most critical in Anantapur's dry climate",
            "Apply Trichoderma viride (4kg/ha) as soil treatment before sowing",
            "Harvest immediately when mature — do not leave standing crop",
            "Rotate with groundnut or redgram the next season"
        ],
        "spray_ok_in_rain": True,
        "heat_note": "Very common in Anantapur's hot dry conditions. Trichoderma at KVK is the best and cheapest preventive. Irrigation during grain fill is the single most impactful intervention.",
        "key_chemical": "trichoderma"
    },
    "Sunflower___healthy": {
        "description": "The sunflower plant appears healthy.",
        "severity": "None",
        "remedy": ["Apply borax (1g/L) spray at bud stage", "Scout for head borer from bud formation", "Ensure adequate moisture during flowering"],
        "spray_ok_in_rain": True, "heat_note": "Boron deficiency is common in Anantapur's red soils — borax spray at bud stage is very important.", "key_chemical": ""
    },

    # 🌾 PADDY
    "Paddy___Blast": {
        "description": "Most devastating rice disease — diamond-shaped grey lesions that can kill crop at neck stage.",
        "severity": "Critical",
        "remedy": [
            "Spray tricyclazole (0.6g/L) at early tillering — do NOT wait",
            "Repeat spray at panicle initiation (PI) stage — this is the MOST CRITICAL spray",
            "Use resistant varieties MTU 1010, Swarna Sub1, or Tellahamsa",
            "Avoid excess nitrogen — use split application (3-4 times)",
            "Drain field every 3-4 days to reduce crop humidity"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Tricyclazole is available at larger agri shops in Anantapur. The PI-stage spray is non-negotiable — missing it causes complete crop loss. Common during kharif paddy in Anantapur's irrigated pockets.",
        "key_chemical": "tricyclazole"
    },
    "Paddy___Brown_spot": {
        "description": "Fungal disease causing oval brown spots — indicates potassium and silicon deficiency in soil.",
        "severity": "Moderate",
        "remedy": [
            "Apply MOP (potassium chloride) fertilizer immediately — correct the deficiency first",
            "Spray mancozeb (2.5g/L) or propiconazole (1ml/L) on leaves",
            "Apply silicate fertilizer if available",
            "Avoid moisture stress during grain filling stage"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "MOP is widely available in Anantapur. Potassium deficiency is common in Anantapur's red soils — apply adequately at transplanting.",
        "key_chemical": "mancozeb"
    },
    "Paddy___Bacterial_leaf_blight": {
        "description": "Bacterial disease causing yellowing from leaf tip downward — leaves turn straw-colored.",
        "severity": "High",
        "remedy": [
            "Spray copper oxychloride (3g/L) + streptocycline (0.5g/L) immediately",
            "Drain field completely — do not flood irrigate while infected",
            "Remove and burn infected plant bunches",
            "Apply split nitrogen — avoid heavy doses that increase susceptibility"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Copper oxychloride is very cheap and available everywhere in Anantapur. Streptocycline at district agri office. Drain field before spraying for better efficacy.",
        "key_chemical": "copper oxychloride"
    },
    "Paddy___Sheath_blight": {
        "description": "Fungal disease causing oval grey lesions on leaf sheath near water level — serious in high-density crops.",
        "severity": "High",
        "remedy": [
            "Spray hexaconazole (1ml/L) or propiconazole (1ml/L) at tillering stage",
            "Reduce plant density — avoid less than 20cm between hills",
            "Drain field to reduce humidity at base of plants",
            "Apply Trichoderma viride at transplanting as preventive biocontrol"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Hexaconazole and propiconazole are available at larger agri shops in Anantapur. Trichoderma at KVK Reddipalli. Draining the field is the cheapest first step.",
        "key_chemical": "hexaconazole"
    },
    "Paddy___healthy": {
        "description": "The paddy plant appears healthy.",
        "severity": "None",
        "remedy": ["Apply zinc sulfate (25 kg/ha) as basal dose — essential for Anantapur soils", "Use leaf colour chart for nitrogen management", "Maintain 2-5cm water during active tillering"],
        "spray_ok_in_rain": True, "heat_note": "Zinc deficiency is very common in Anantapur paddy — apply zinc sulfate as basal dose without fail.", "key_chemical": ""
    },

    # 🍎 POMEGRANATE
    "Pomegranate___Bacterial_blight": {
        "description": "Bacterial disease causing water-soaked spots on leaves and dark cankers on branches — most damaging pomegranate disease in Anantapur.",
        "severity": "High",
        "remedy": [
            "Spray copper hydroxide (2g/L) + streptocycline (200ppm) before southwest monsoon",
            "Remove and burn all infected branches immediately",
            "Use drip irrigation only — overhead water spreads bacteria rapidly",
            "Apply bordeaux mixture (1%) on trunk during dormancy (Dec-Jan)",
            "Disinfect all pruning tools with 10% bleach between each tree"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Copper hydroxide is widely available in Anantapur. Streptocycline at district agri office. The pre-monsoon spray is the single most important action for pomegranate in Anantapur.",
        "key_chemical": "copper hydroxide"
    },
    "Pomegranate___Cercospora_leaf_spot": {
        "description": "Fungal disease causing circular brown spots with grey centres — causes premature leaf fall.",
        "severity": "Moderate",
        "remedy": [
            "Spray carbendazim (1g/L) or copper oxychloride (3g/L) at first appearance",
            "Remove and destroy all fallen infected leaves — burn them",
            "Prune selectively after harvest to open up canopy"
        ],
        "spray_ok_in_rain": False,
        "heat_note": "Both carbendazim and copper oxychloride are cheaply and widely available in Anantapur. Use carbendazim for systemic action.",
        "key_chemical": "carbendazim"
    },
    "Pomegranate___healthy": {
        "description": "The pomegranate plant appears healthy.",
        "severity": "None",
        "remedy": ["Apply potassium-rich fertilizer at fruit development", "Use drip irrigation — pomegranate is very drought tolerant but needs moisture at fruit set", "Prune after harvest to prevent bacterial blight next season"],
        "spray_ok_in_rain": True, "heat_note": "Pomegranate is ideal for Anantapur's dry climate. Drip irrigation gives best results with very little water.", "key_chemical": ""
    }
}

CLASS_NAMES = list(disease_remedies.keys())

# ─── Image Transform ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── Load Model ──────────────────────────────────────────────────────────────
def load_model():
    m = models.resnet50(pretrained=False)
    
    # Changed output features from len(CLASS_NAMES) to exactly 16
    m.fc = nn.Sequential(
        nn.Identity(), 
        nn.Linear(m.fc.in_features, 16) 
    )
    
    m.load_state_dict(torch.load("model.pth", map_location="cpu"))
    m.eval()
    return m

model = load_model()



# ─── Weather Fetch ────────────────────────────────────────────────────────────
def get_weather():
    """
    Fetches live weather for Anantapur using OpenWeatherMap free API.
    Set WEATHER_API_KEY in HuggingFace Space secrets.
    Returns a dict with weather data or a fallback dict if API key not set.
    """
    api_key = os.environ.get("WEATHER_API_KEY", "")
    if not api_key:
        # Fallback: use Anantapur's typical values so app still works without key
        return {
            "temp": 35, "humidity": 40, "rain_mm": 0,
            "wind_kmh": 10, "description": "Clear sky (demo — add API key for live data)",
            "source": "fallback"
        }
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={LAT}&lon={LON}&appid={api_key}&units=metric")
        r = requests.get(url, timeout=5)
        d = r.json()
        return {
            "temp":        round(d["main"]["temp"], 1),
            "humidity":    d["main"]["humidity"],
            "rain_mm":     d.get("rain", {}).get("1h", 0),
            "wind_kmh":    round(d["wind"]["speed"] * 3.6, 1),
            "description": d["weather"][0]["description"].capitalize(),
            "source":      "live"
        }
    except Exception:
        return {
            "temp": 35, "humidity": 40, "rain_mm": 0,
            "wind_kmh": 10, "description": "Weather data unavailable",
            "source": "fallback"
        }

# ─── Weather Advisory Engine ─────────────────────────────────────────────────
def build_weather_advisory(wx, info, disease_name):
    """
    Reads current weather and builds spray safety + urgency advisories
    tailored to Anantapur's climate and the detected disease.
    """
    notes = []
    spray_ok_in_rain = info.get("spray_ok_in_rain", False)
    severity         = info.get("severity", "None")
    heat_note        = info.get("heat_note", "")
    key_chem         = info.get("key_chemical", "")

    # ── Spray timing ──
    if wx["rain_mm"] > 0:
        if spray_ok_in_rain:
            notes.append("🌧️ It is raining — this treatment can still be applied, but wait for rain to stop for better absorption.")
        else:
            notes.append("🌧️ RAIN ALERT: Do NOT spray contact fungicides right now — they will wash off within 30 minutes. Wait for at least 6 dry hours after rain stops before spraying.")
    elif wx["humidity"] > 65:
        notes.append(f"💧 HIGH HUMIDITY ({wx['humidity']}%): Disease spread risk is elevated. Spray as soon as possible — today if severity is High or Critical.")
    elif wx["humidity"] < 35:
        notes.append(f"🌵 LOW HUMIDITY ({wx['humidity']}%): Typical dry Anantapur weather. Disease pressure from fungal diseases is lower, but spider mites and thrips risk is higher.")

    # ── Heat advisory ──
    if wx["temp"] >= 38:
        notes.append(f"🔥 EXTREME HEAT ({wx['temp']}°C): Spray ONLY before 8:00 AM or after 5:30 PM. Do NOT use sulfur-based fungicides — they burn leaves above 35°C. Use copper or systemic fungicides instead.")
    elif wx["temp"] >= 32:
        notes.append(f"☀️ HOT DAY ({wx['temp']}°C): Spray in early morning (before 9 AM) for best results. Avoid midday spraying — chemicals evaporate quickly and can scorch leaves.")
    else:
        notes.append(f"🌤️ GOOD SPRAYING TEMPERATURE ({wx['temp']}°C): Current conditions are suitable for spraying.")

    # ── Wind advisory ──
    if wx["wind_kmh"] > 15:
        notes.append(f"💨 WIND WARNING ({wx['wind_kmh']} km/h): Wind is too high for effective spraying — chemicals drift away from the crop. Wait for wind to drop below 10 km/h (usually early morning or evening).")
    elif wx["wind_kmh"] > 8:
        notes.append(f"🍃 MODERATE WIND ({wx['wind_kmh']} km/h): Use a low-pressure sprayer and shield the nozzle. Early morning is calmer.")

    # ── Disease + weather interaction ──
    if severity in ["High", "Critical"] and wx["humidity"] > 60:
        notes.append("⚠️ HIGH HUMIDITY + SEVERE DISEASE: These conditions will spread infection rapidly to neighbouring plants. Treat today — not tomorrow.")
    if "Rust" in disease_name and wx["humidity"] > 55:
        notes.append("🍂 RUST + HUMIDITY: Rust spreads explosively in humid conditions. Check all neighbouring groundnut plots — they likely need treatment too.")
    if "Sigatoka" in disease_name and wx["rain_mm"] > 2:
        notes.append("🍌 SIGATOKA + RAIN: Sigatoka spreads by rain splash. Remove infected leaves immediately before spraying to reduce spore load.")
    if "wilt" in disease_name.lower() and wx["rain_mm"] > 5:
        notes.append("🌊 WILT + RAIN: Wilt diseases spread fastest through waterlogged soil. Improve drainage urgently. Do not enter field with wet boots — you spread it.")

    # ── Local availability ──
    if key_chem and key_chem in LOCAL:
        notes.append(f"🏪 LOCAL AVAILABILITY — {key_chem.capitalize()}: {LOCAL[key_chem]}")

    # ── heat_note ──
    if heat_note:
        notes.append(f"📍 LOCAL TIP: {heat_note}")

    return "\n\n".join(notes)

# ─── Severity Display ─────────────────────────────────────────────────────────
def severity_display(s):
    return {
        "None":     "✅ None — Plant is healthy",
        "Low":      "🟡 Low — Monitor closely",
        "Moderate": "🟠 Moderate — Treat within 3-5 days",
        "High":     "🔴 High — Treat TODAY",
        "Critical": "🚨 Critical — Act NOW and contact KVK"
    }.get(s, s)

# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload an image.", "", "", "", "No weather data yet.", ""

    # 1. Get live weather
    wx = get_weather()

    # 2. Format weather panel
    src_tag  = "🌐 Live" if wx["source"] == "live" else "📊 Demo (add API key for live)"
    wx_panel = (
        f"📍 {CITY}  |  {src_tag}\n"
        f"🌡️ {wx['temp']}°C   💧 Humidity: {wx['humidity']}%   "
        f"🌧️ Rain: {wx['rain_mm']} mm/hr   💨 Wind: {wx['wind_kmh']} km/h\n"
        f"☁️ {wx['description']}"
    )

    # 3. Run model
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs    = model(img_tensor)
        probs      = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item()
    disease    = CLASS_NAMES[pred.item()]
    info       = disease_remedies.get(disease, {})

    # 4. Low confidence fallback
    if confidence < 0.75:
        return (
            f"⚠️ Low Confidence ({confidence*100:.1f}%)",
            "Model is uncertain — this may be a crop not in the supported list, or the image is unclear. Try a clearer, closer photo of the leaf.",
            "❓ Unknown",
            "",
            wx_panel,
            "Upload a clearer image, or check the supported crop list on the left."
        )

    # 5. Build outputs
    disease_display = disease.replace("_", " ").replace("  ", " ")
    description     = info.get("description", "")
    severity        = severity_display(info.get("severity", ""))
    remedy_text     = "\n".join([f"• {s}" for s in info.get("remedy", [])])
    advisory        = build_weather_advisory(wx, info, disease)

    return (
        f"🌿 {disease_display}   ({confidence*100:.1f}% confidence)",
        description,
        severity,
        remedy_text,
        wx_panel,
        advisory
    )

# ─── Gradio UI ────────────────────────────────────────────────────────────────
supported_text = " · ".join(SUPPORTED_CROPS)

with gr.Blocks(title="🌿 Crop Disease Detector — Anantapur", theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"""
    # 🌿 Crop Disease Detector — Anantapur Edition
    **Weather-aware disease diagnosis and remedy advisor for Anantapur, Andhra Pradesh 🇮🇳**
    Remedies are adjusted to live weather conditions, Anantapur's semi-arid climate, and local chemical availability.

    **Supported crops:** {supported_text}
    """)

    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload Leaf Image")
            submit_btn  = gr.Button("🔍 Detect Disease & Get Weather-Aware Advice", variant="primary")
            gr.Markdown("""
            **Tips for best results:**
            - 📸 Take photo in bright natural light
            - 🍃 Fill frame with the diseased leaf
            - 🚫 Avoid blurry or distant photos
            - ✅ One leaf per photo works best
            """)

        # Right column
        with gr.Column(scale=2):
            weather_output  = gr.Textbox(label="🌤️ Live Weather — Anantapur District", lines=3)
            disease_output  = gr.Textbox(label="🦠 Detected Disease")
            desc_output     = gr.Textbox(label="📋 What is this disease?", lines=3)
            severity_output = gr.Textbox(label="⚠️ Severity & Urgency")
            remedy_output   = gr.Textbox(label="💊 Treatment Steps", lines=7)
            advisory_output = gr.Textbox(label="🌦️ Weather-Adjusted Advice + Local Availability", lines=12)

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[disease_output, desc_output, severity_output, remedy_output, weather_output, advisory_output]
    )

    gr.Markdown("""
    ---
    ⚠️ **Disclaimer:** Educational use only. Always confirm with a certified agronomist for critical decisions.

    📍 **Anantapur Krishi Vigyan Kendra (KVK):** [kvkreddipalli-angrau.org](https://www.kvkreddipalli-angrau.org) · 📞 08554-255560
    🌾 **District Agriculture Office, Anantapur:** 📞 08554-274400
    """)

demo.launch()
