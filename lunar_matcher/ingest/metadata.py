"""PDS/ISSDC metadata parser for Chandrayaan-2 planetary data products."""

import re
import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional


class MalformedMetadataError(ValueError):
    """Raised when an ISSDC/PDS metadata sidecar is missing critical fields or malformed."""
    pass


def parse_lbl_text(text: str) -> Dict[str, str]:
    """Parse standard PDS3 / ISSDC key = value text format."""
    data = {}
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("/*") or line == "END":
            continue
        if "=" in line:
            parts = line.split("=", 1)
            key = parts[0].strip().upper()
            val = parts[1].strip().strip('"').strip("'")
            data[key] = val
    return data


def parse_issdc_metadata(sidecar_path: str) -> Dict[str, Any]:
    """
    Parse an ISSDC/PDS sidecar (.lbl or .xml) and extract georeferencing and illumination metadata.
    
    Raises:
        MalformedMetadataError: If required fields (corners or projection) are missing.
        FileNotFoundError: If the sidecar does not exist.
    """
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Metadata sidecar not found: {sidecar_path}")

    ext = os.path.splitext(sidecar_path)[1].lower()
    
    if ext == ".xml":
        return _parse_xml_metadata(sidecar_path)
    elif ext in [".lbl", ".txt", ".pds"]:
        return _parse_lbl_metadata(sidecar_path)
    else:
        # Attempt LBL parse first, then XML fallback
        try:
            return _parse_lbl_metadata(sidecar_path)
        except Exception:
            return _parse_xml_metadata(sidecar_path)


def _parse_lbl_metadata(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    raw = parse_lbl_text(text)
    
    # Check for required coordinates
    # ISSDC products commonly use UPPER_LEFT_LATITUDE or MINIMUM_LATITUDE / MAXIMUM_LATITUDE
    corners = {}
    
    # Corner extraction
    has_explicit_corners = False
    if "UPPER_LEFT_LATITUDE" in raw and "LOWER_RIGHT_LATITUDE" in raw:
        try:
            corners = {
                "ul_lat": float(raw["UPPER_LEFT_LATITUDE"]),
                "ul_lon": float(raw["UPPER_LEFT_LONGITUDE"]),
                "lr_lat": float(raw["LOWER_RIGHT_LATITUDE"]),
                "lr_lon": float(raw["LOWER_RIGHT_LONGITUDE"]),
                "ur_lat": float(raw.get("UPPER_RIGHT_LATITUDE", raw["UPPER_LEFT_LATITUDE"])),
                "ur_lon": float(raw.get("UPPER_RIGHT_LONGITUDE", raw["LOWER_RIGHT_LONGITUDE"])),
                "ll_lat": float(raw.get("LOWER_LEFT_LATITUDE", raw["LOWER_RIGHT_LATITUDE"])),
                "ll_lon": float(raw.get("LOWER_LEFT_LONGITUDE", raw["UPPER_LEFT_LONGITUDE"])),
            }
            has_explicit_corners = True
        except (ValueError, KeyError) as e:
            raise MalformedMetadataError(f"Malformed corner coordinate numeric values in {path}: {e}")
            
    elif "MAXIMUM_LATITUDE" in raw and "MINIMUM_LATITUDE" in raw:
        try:
            corners = {
                "ul_lat": float(raw["MAXIMUM_LATITUDE"]),
                "ul_lon": float(raw.get("WESTERNMOST_LONGITUDE", raw.get("MINIMUM_LONGITUDE", 0))),
                "lr_lat": float(raw["MINIMUM_LATITUDE"]),
                "lr_lon": float(raw.get("EASTERNMOST_LONGITUDE", raw.get("MAXIMUM_LONGITUDE", 0))),
                "ur_lat": float(raw["MAXIMUM_LATITUDE"]),
                "ur_lon": float(raw.get("EASTERNMOST_LONGITUDE", raw.get("MAXIMUM_LONGITUDE", 0))),
                "ll_lat": float(raw["MINIMUM_LATITUDE"]),
                "ll_lon": float(raw.get("WESTERNMOST_LONGITUDE", raw.get("MINIMUM_LONGITUDE", 0))),
            }
            has_explicit_corners = True
        except (ValueError, KeyError) as e:
            raise MalformedMetadataError(f"Malformed bounding box coordinates in {path}: {e}")

    if not has_explicit_corners:
        raise MalformedMetadataError(
            f"Missing required coordinate fields in metadata: {path}. "
            f"Expected UPPER_LEFT_LATITUDE/LONGITUDE or MAXIMUM_LATITUDE/MINIMUM_LATITUDE."
        )

    # Projection info
    projection_name = raw.get("MAP_PROJECTION_TYPE", raw.get("PROJECTION", "Equirectangular"))
    proj_params = raw.get("MAP_PROJECTION_NAME", raw.get("COORDINATE_SYSTEM_NAME", "Moon_2000"))

    # Sun illumination info
    sun_elevation = None
    sun_azimuth = None
    incidence_angle = None
    
    for k, v in raw.items():
        if "SOLAR_INCIDENCE_ANGLE" in k or "INCIDENCE_ANGLE" in k:
            try:
                incidence_angle = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", v)[0])
            except Exception:
                pass
        elif "SUN_ELEVATION" in k or "SOLAR_ELEVATION" in k:
            try:
                sun_elevation = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", v)[0])
            except Exception:
                pass
        elif "SUN_AZIMUTH" in k or "SOLAR_AZIMUTH" in k:
            try:
                sun_azimuth = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", v)[0])
            except Exception:
                pass

    # If incidence angle not directly given but sun elevation is:
    if incidence_angle is None and sun_elevation is not None:
        incidence_angle = max(0.0, 90.0 - sun_elevation)
    elif sun_elevation is None and incidence_angle is not None:
        sun_elevation = max(0.0, 90.0 - incidence_angle)

    sensor = raw.get("INSTRUMENT_ID", raw.get("INSTRUMENT_NAME", "UNKNOWN")).upper()
    product_id = raw.get("PRODUCT_ID", os.path.basename(path))

    return {
        "format": "PDS3_LBL",
        "product_id": product_id,
        "sensor": sensor,
        "corners": corners,
        "projection_name": projection_name,
        "proj_params": proj_params,
        "sun_elevation_deg": sun_elevation,
        "sun_azimuth_deg": sun_azimuth,
        "solar_incidence_angle_deg": incidence_angle,
        "raw_dict": raw,
    }


def _parse_xml_metadata(path: str) -> Dict[str, Any]:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        raise MalformedMetadataError(f"Failed to parse XML metadata {path}: {e}")

    # Remove namespace prefixes for easier querying
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    corners = {}
    ul_lat = root.find(".//upper_left_latitude")
    ul_lon = root.find(".//upper_left_longitude")
    lr_lat = root.find(".//lower_right_latitude")
    lr_lon = root.find(".//lower_right_longitude")

    if ul_lat is not None and ul_lon is not None and lr_lat is not None and lr_lon is not None:
        try:
            corners = {
                "ul_lat": float(ul_lat.text),
                "ul_lon": float(ul_lon.text),
                "lr_lat": float(lr_lat.text),
                "lr_lon": float(lr_lon.text),
                "ur_lat": float(root.findtext(".//upper_right_latitude", ul_lat.text)),
                "ur_lon": float(root.findtext(".//upper_right_longitude", lr_lon.text)),
                "ll_lat": float(root.findtext(".//lower_left_latitude", lr_lat.text)),
                "ll_lon": float(root.findtext(".//lower_left_longitude", ul_lon.text)),
            }
        except ValueError as e:
            raise MalformedMetadataError(f"Invalid numeric coordinate in XML {path}: {e}")
    else:
        # Try westernmost/easternmost
        w_lon = root.find(".//westernmost_longitude")
        e_lon = root.find(".//easternmost_longitude")
        s_lat = root.find(".//minimum_latitude")
        n_lat = root.find(".//maximum_latitude")
        if w_lon is not None and e_lon is not None and s_lat is not None and n_lat is not None:
            corners = {
                "ul_lat": float(n_lat.text),
                "ul_lon": float(w_lon.text),
                "lr_lat": float(s_lat.text),
                "lr_lon": float(e_lon.text),
                "ur_lat": float(n_lat.text),
                "ur_lon": float(e_lon.text),
                "ll_lat": float(s_lat.text),
                "ll_lon": float(w_lon.text),
            }
        else:
            raise MalformedMetadataError(
                f"Missing required coordinate nodes in XML metadata: {path}. "
                f"Expected upper_left_latitude/longitude or bounding box."
            )

    proj_elem = root.find(".//projection_name") or root.find(".//map_projection_type")
    proj_name = proj_elem.text if proj_elem is not None else "Equirectangular"

    inc_elem = root.find(".//solar_incidence_angle") or root.find(".//incidence_angle")
    sun_elev_elem = root.find(".//sun_elevation") or root.find(".//solar_elevation")
    sun_az_elem = root.find(".//sun_azimuth") or root.find(".//solar_azimuth")

    incidence_angle = float(inc_elem.text) if inc_elem is not None and inc_elem.text else None
    sun_elevation = float(sun_elev_elem.text) if sun_elev_elem is not None and sun_elev_elem.text else None
    sun_azimuth = float(sun_az_elem.text) if sun_az_elem is not None and sun_az_elem.text else None

    if incidence_angle is None and sun_elevation is not None:
        incidence_angle = max(0.0, 90.0 - sun_elevation)
    elif sun_elevation is None and incidence_angle is not None:
        sun_elevation = max(0.0, 90.0 - incidence_angle)

    sensor_elem = root.find(".//instrument_name") or root.find(".//instrument_id")
    sensor = sensor_elem.text.upper() if sensor_elem is not None and sensor_elem.text else "UNKNOWN"

    return {
        "format": "PDS4_XML",
        "product_id": os.path.basename(path),
        "sensor": sensor,
        "corners": corners,
        "projection_name": proj_name,
        "sun_elevation_deg": sun_elevation,
        "sun_azimuth_deg": sun_azimuth,
        "solar_incidence_angle_deg": incidence_angle,
    }
