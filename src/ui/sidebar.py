import streamlit as st
from src.config import LOCATION_REGISTRY, DEFAULT_LOCATION

def render_sidebar() -> tuple[str, str]:
    """Renders the sidebar municipality selector and returns (target_location_key, display_title)."""
    st.sidebar.header("📍 Target Profile Setup")
    location_options = sorted([loc.replace("_", " ").title() for loc in LOCATION_REGISTRY.keys()])

    default_title = DEFAULT_LOCATION.replace("_", " ").title()
    default_index = location_options.index(default_title) if default_title in location_options else 0

    selected_title = st.sidebar.selectbox(
        "Select Municipality / City:",
        location_options,
        index=default_index
    )
    target_location = selected_title.lower().replace(" ", "_")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Provincial Coordinates:**")
    st.sidebar.text(f"Latitude:  {LOCATION_REGISTRY[target_location]['lat']}")
    st.sidebar.text(f"Longitude: {LOCATION_REGISTRY[target_location]['lon']}")
    st.sidebar.markdown("---")
    st.sidebar.info("☁️ Cloud Logger: Read-Only Mode (3-Hour Scheduled Cron)")

    return target_location, selected_title
