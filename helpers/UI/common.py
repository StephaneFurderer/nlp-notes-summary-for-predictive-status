import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple, List

from helpers.UI.sidebar import (
    initialize_sidebar,
    advanced_sidebar,
    test_if_claim_number_is_valid,
)
from helpers.functions.claims_utils import (
    read_transformed_claims_data_from_parquet,
    read_periods_data,
)


@st.cache_data(show_spinner=False)
def _load_shared_data(extraction_date: str) -> Dict[str, Any]:
    df_raw_txn, _, _, _, df_raw_final, _, _, _ = read_transformed_claims_data_from_parquet(extraction_date)
    df_raw_txn_to_periods, _, _, _ = read_periods_data(extraction_date)

    # merge the 
    return {
        "df_raw_txn": df_raw_txn,
        "df_raw_final": df_raw_final,
        "df_raw_txn_to_periods": df_raw_txn_to_periods,
    }


def get_shared_state() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build common sidebar and return raw and filtered data.

    Returns:
        raw: dict with df_raw_txn, df_raw_final, df_raw_txn_to_periods
        filtered: dict with keys: data{...} and filters{extraction_date,cause,status,claim_number}
    """
    extraction_date = initialize_sidebar()

    raw = _load_shared_data(extraction_date)

    [df_txn_f, df_final_f, df_periods_f], cause, status, claim_number = advanced_sidebar([
        raw["df_raw_txn"], raw["df_raw_final"], raw["df_raw_txn_to_periods"],
    ])

    filtered = {
        "data": {
            "df_raw_txn": df_txn_f,
            "df_raw_final": df_final_f,
            "df_raw_txn_to_periods": df_periods_f,
        },
        "filters": {
            "extraction_date": extraction_date,
            "cause": cause,
            "status": status,
            "claim_number": claim_number,
        },
    }

    return raw, filtered