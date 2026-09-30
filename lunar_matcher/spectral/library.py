"""Mineral spectral library and diagnostic absorption band reference positions.

Citations:
- Pieters, C. M. (1993). Compositional diversity and geology of the lunar crust:
  Contemplating the Moon. Remote Geochemical Analysis: Elemental and Mineralogical Composition.
- Clenet, H., et al. (2011). A new systematic approach for characterizing and quantifying
  absorption bands in spectral data: Application to lunar pyroxenes. Icarus, 213(2), 404-422.
- Clark, R. N., et al. (2007). USGS digital spectral library splib06a. US Geological Survey.
- RELAB Spectral Database, Brown University Keck/NASA Reflectance Experiment Laboratory.
"""

from typing import Dict, Any, Optional

# Reference mineral definitions with characteristic absorption bands and continuum slope
MINERAL_LIBRARY: Dict[str, Dict[str, Any]] = {
    "low_ca_pyroxene": {
        "name": "Low-Ca Pyroxene (Orthopyroxene)",
        "mineral_group": "Pyroxene",
        "description": "Common in lunar highlands norites and lower crustal excavated crater walls.",
        "diagnostic_bands": {
            "band1": {"center_nm": 930.0, "range_nm": (890.0, 960.0), "min_depth": 0.04},
            "band2": {"center_nm": 1900.0, "range_nm": (1800.0, 2000.0), "min_depth": 0.05},
        },
        "band_area_ratio_expected": (0.8, 1.8),
        "color_hex": "#3b82f6",  # Blue
    },
    "high_ca_pyroxene": {
        "name": "High-Ca Pyroxene (Clinopyroxene)",
        "mineral_group": "Pyroxene",
        "description": "Dominant in lunar mare basalts and mafic mare flows (augite/diopside).",
        "diagnostic_bands": {
            "band1": {"center_nm": 1015.0, "range_nm": (980.0, 1060.0), "min_depth": 0.04},
            "band2": {"center_nm": 2120.0, "range_nm": (2050.0, 2250.0), "min_depth": 0.05},
        },
        "band_area_ratio_expected": (0.9, 2.0),
        "color_hex": "#10b981",  # Green
    },
    "olivine": {
        "name": "Olivine-rich (Dunite / Troctolite)",
        "mineral_group": "Olivine",
        "description": "Mantle-derived or deep crustal material excavated by large impact basins.",
        "diagnostic_bands": {
            "band1": {"center_nm": 1050.0, "range_nm": (1020.0, 1100.0), "min_depth": 0.06},
            "band2": None,  # Olivine lacks 2-micron pyroxene absorption
        },
        "band_area_ratio_expected": (0.0, 0.3),
        "color_hex": "#eab308",  # Amber / Yellow
    },
    "plagioclase": {
        "name": "Plagioclase-rich (Anorthosite)",
        "mineral_group": "Feldspar",
        "description": "Primary constituent of the lunar primordial magma ocean floatation crust.",
        "diagnostic_bands": {
            "band1": {"center_nm": 1250.0, "range_nm": (1200.0, 1320.0), "min_depth": 0.015},
            "band2": None,
        },
        "color_hex": "#a855f7",  # Purple
    },
    "spinel": {
        "name": "Spinel-like (Mg-Al Spinel)",
        "mineral_group": "Oxide",
        "description": "Rare deep-seated pink spinel anorthosite (PSA) lacking 1 um mafic absorption.",
        "diagnostic_bands": {
            "band1": None,
            "band2": {"center_nm": 2020.0, "range_nm": (1950.0, 2100.0), "min_depth": 0.04},
        },
        "color_hex": "#ec4899",  # Pink
    },
    "featureless_dark": {
        "name": "Featureless / Dark (Ilmenite-rich or Mature)",
        "mineral_group": "Oxide/Regolith",
        "description": "High-Ti mature mare regolith or ilmenite-rich basalt with suppressed absorption.",
        "diagnostic_bands": {},
        "color_hex": "#64748b",  # Slate
    },
    "unclassified": {
        "name": "Unclassified Mixture",
        "mineral_group": "Mixed",
        "description": "Complex regolith mixture with weak or ambiguous absorption signatures.",
        "diagnostic_bands": {},
        "color_hex": "#475569",  # Dark slate
    },
}


def get_mineral_reference(key: str) -> Optional[Dict[str, Any]]:
    """Retrieve reference mineral criteria by canonical key."""
    return MINERAL_LIBRARY.get(key)
