"""
Owner: Person A

Normalize business_name and business_address so downstream blocking/features
work off consistent text. Keep this country-agnostic — France appears only in
test, so nothing here should branch on a hardcoded {US, India} set.

TODO:
- lowercase, strip punctuation, collapse whitespace
- expand legal-suffix abbreviations (Pvt/Private, Ltd/Limited, Corp/Corporation)
- expand address abbreviations (Rd/Road, St/Street, Apt/Apartment)
- extract structured sub-tokens where present (pin/zip code, street number) via regex
- keep both raw and normalized columns; some similarity features want raw text
"""
import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a source DataFrame with at least [entity_id, business_name,
    business_address, country].
    Output: same DataFrame with added columns, e.g. name_norm, address_norm,
    address_pin (nullable), address_street_number (nullable).
    """
    out = df.copy()
    # TODO: implement
    out["name_norm"] = out["business_name"]
    out["address_norm"] = out["business_address"]
    return out
