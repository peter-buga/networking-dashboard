"""
Configuration settings for the topology visualization service.
"""
import os


def get_int_env(key: str, default: int) -> int:
    """Get integer value from environment variable with fallback to default."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def get_float_env(key: str, default: float) -> float:
    """Get float value from environment variable with fallback to default."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


# Topology visualization configuration
TOPOLOGY_CONFIG = {
    # Node styling
    'node_size': get_int_env('TOPOLOGY_NODE_SIZE', 2500),

    # Font sizing
    'font_size_base': get_int_env('TOPOLOGY_FONT_SIZE_BASE', 10),
    'font_size_min': get_int_env('TOPOLOGY_FONT_SIZE_MIN', 8),
    'font_size_max': get_int_env('TOPOLOGY_FONT_SIZE_MAX', 14),

    # Label positioning
    'edge_label_offset': get_float_env('TOPOLOGY_EDGE_LABEL_OFFSET', 0.1),
    'ip_label_distance': get_float_env('TOPOLOGY_IP_LABEL_DISTANCE', 0.2),

    # Text styling
    'text_padding': get_float_env('TOPOLOGY_TEXT_PADDING', 0.3),

    # Layout
    'figure_margin': get_float_env('TOPOLOGY_FIGURE_MARGIN', 0.2),
}

# File paths
DEFAULT_TOPOLOGY_FILE = os.environ.get('DEFAULT_TOPOLOGY_FILE', '/app/base_files/testnet.py')

# Image generation settings
IMAGE_CONFIG = {
    'default_width': get_int_env('TOPOLOGY_IMAGE_WIDTH', 1400),
    'default_height': get_int_env('TOPOLOGY_IMAGE_HEIGHT', 500),
    'dpi': get_int_env('TOPOLOGY_IMAGE_DPI', 200),
    'format': os.environ.get('TOPOLOGY_IMAGE_FORMAT', 'png'),
}