"""Central configuration: paths, factory master data, product->factory map, assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "Nassau Candy Distributor.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Factories (from the project brief)
# --------------------------------------------------------------------------- #
FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

# Current (historical) product -> factory assignment (from the project brief)
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

SHIP_MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]

# Fixed colour per factory (categorical, fixed order - never re-assigned)
FACTORY_COLORS = {
    "Lot's O' Nuts": "#2a6ebb",
    "Wicked Choccy's": "#d9822b",
    "Sugar Shack": "#3a9d8f",
    "Secret Factory": "#8c5fb3",
    "The Other Factory": "#c4506a",
}

# --------------------------------------------------------------------------- #
# Destination coordinates: approximate state centroids (US) and approximate
# population-weighted centres (Canadian provinces). Used only for distance.
# --------------------------------------------------------------------------- #
STATE_COORDS = {
    "Alabama": (32.81, -86.79), "Arizona": (34.29, -111.66), "Arkansas": (34.90, -92.44),
    "California": (37.18, -119.47), "Colorado": (39.00, -105.55), "Connecticut": (41.62, -72.73),
    "Delaware": (38.99, -75.51), "District of Columbia": (38.91, -77.01), "Florida": (28.63, -82.45),
    "Georgia": (32.65, -83.43), "Idaho": (44.35, -114.61), "Illinois": (40.04, -89.20),
    "Indiana": (39.89, -86.28), "Iowa": (42.07, -93.50), "Kansas": (38.49, -98.38),
    "Kentucky": (37.53, -85.30), "Louisiana": (31.07, -92.00), "Maine": (45.37, -69.24),
    "Maryland": (39.05, -76.79), "Massachusetts": (42.26, -71.81), "Michigan": (44.35, -85.41),
    "Minnesota": (46.28, -94.31), "Mississippi": (32.74, -89.67), "Missouri": (38.36, -92.46),
    "Montana": (47.05, -109.64), "Nebraska": (41.54, -99.80), "Nevada": (39.33, -116.63),
    "New Hampshire": (43.68, -71.58), "New Jersey": (40.19, -74.67), "New Mexico": (34.41, -106.11),
    "New York": (42.95, -75.53), "North Carolina": (35.56, -79.39), "North Dakota": (47.45, -100.47),
    "Ohio": (40.29, -82.79), "Oklahoma": (35.59, -97.49), "Oregon": (43.94, -120.56),
    "Pennsylvania": (40.89, -77.80), "Rhode Island": (41.68, -71.51), "South Carolina": (33.91, -80.90),
    "South Dakota": (44.44, -100.23), "Tennessee": (35.86, -86.35), "Texas": (31.48, -99.33),
    "Utah": (39.31, -111.67), "Vermont": (44.07, -72.67), "Virginia": (37.52, -78.85),
    "Washington": (47.38, -120.45), "West Virginia": (38.64, -80.62), "Wisconsin": (44.62, -89.99),
    "Wyoming": (43.00, -107.55),
    # Canada
    "Alberta": (52.00, -113.50), "British Columbia": (49.80, -122.50), "Manitoba": (49.90, -97.20),
    "New Brunswick": (46.10, -66.20), "Newfoundland and Labrador": (47.60, -53.00),
    "Nova Scotia": (44.90, -63.50), "Ontario": (44.00, -79.80), "Prince Edward Island": (46.40, -63.30),
    "Quebec": (46.00, -73.50), "Saskatchewan": (51.50, -105.50),
}

# --------------------------------------------------------------------------- #
# Modelling assumptions (adjustable in the Streamlit sidebar)
# The dataset has no freight-cost or transit-time field, so these are explicit,
# documented assumptions - not estimates learned from data.
# --------------------------------------------------------------------------- #
DEFAULT_ASSUMPTIONS = {
    # extra transit days per 1,000 miles of factory->customer distance (Standard Class basis)
    "days_per_1000mi": 0.6,
    # freight cost in USD per unit per 1,000 miles (Standard Class basis)
    "freight_usd_per_unit_1000mi": 0.08,
    # profit-drop (as % of gross profit) beyond which a reassignment is flagged high-risk
    "profit_alert_pct": -2.0,
    # minimum scenario confidence (0-100) below which a reassignment is flagged high-risk
    "min_confidence": 50.0,
}

# Ship-mode multipliers on the transit-time and freight-cost assumptions
MODE_TRANSIT_MULT = {"Same Day": 0.3, "First Class": 0.6, "Second Class": 0.8, "Standard Class": 1.0}
MODE_COST_MULT = {"Same Day": 2.5, "First Class": 1.7, "Second Class": 1.3, "Standard Class": 1.0}
