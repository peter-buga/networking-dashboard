"""
Utility functions for topology visualization and text positioning.
"""
import math
from typing import Dict, List, Tuple

from .config import TOPOLOGY_CONFIG as CONFIG


def calculate_text_bounds(text: str, x: float, y: float, fontsize: int, padding_multiplier: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Calculate text bounds for collision detection with padding.

    Args:
        text: The text string
        x: X coordinate of text center
        y: Y coordinate of text center
        fontsize: Font size in points
        padding_multiplier: Multiplier for additional padding around text

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) bounding box coordinates
    """
    # More accurate character sizing based on typical font metrics
    char_width = fontsize * 0.8 / 100  # Increased from 0.6 to 0.8 for better accuracy
    char_height = fontsize * 1.4 / 100  # Increased from 1.2 to 1.4 for line height

    lines = text.split('\n')
    max_width = max(len(line) for line in lines) if lines else 0
    height = len(lines)

    # Calculate base dimensions
    width = max_width * char_width
    total_height = height * char_height

    # Add padding based on the configured text padding and multiplier
    padding = CONFIG['text_padding'] * padding_multiplier / 100
    width += padding * 2
    total_height += padding * 2

    return (x - width/2, y - total_height/2, x + width/2, y + total_height/2)


def boxes_overlap(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> bool:
    """
    Check if two text bounding boxes overlap.

    Args:
        box1: First bounding box (x_min, y_min, x_max, y_max)
        box2: Second bounding box (x_min, y_min, x_max, y_max)

    Returns:
        True if boxes overlap, False otherwise
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def get_edge_midpoint_and_rotation(pos: Dict[str, Tuple[float, float]], node1: str, node2: str) -> Tuple[float, float, float]:
    """
    Calculate edge midpoint and rotation angle for label placement.

    Args:
        pos: Dictionary mapping node names to (x, y) coordinates
        node1: First node name
        node2: Second node name

    Returns:
        Tuple of (mid_x, mid_y, rotation_angle) where rotation is in degrees
    """
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

    # Calculate rotation angle
    angle = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle)

    # Keep text readable (don't rotate beyond ±90 degrees)
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return mid_x, mid_y, angle_deg


def find_optimal_ip_position(pos: Dict[str, Tuple[float, float]], host: str,
                           occupied_boxes: List[Tuple[float, float, float, float]],
                           text: str, fontsize: int) -> Tuple[float, float]:
    """
    Find optimal position for IP label below a node with enhanced spacing.

    Args:
        pos: Dictionary mapping node names to (x, y) coordinates
        host: Host node name
        occupied_boxes: List of already occupied text bounding boxes
        text: Text content for the IP label
        fontsize: Font size for the label

    Returns:
        Tuple of (x, y) coordinates for optimal label placement below the node
    """
    x, y = pos[host]
    base_distance = CONFIG['ip_label_distance']

    # Try positions below the node with increasing distance
    # Start with larger multipliers to create more space from edge labels
    for multiplier in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        test_x = x
        test_y = y - base_distance * multiplier

        # Use enhanced padding for IP labels to prevent close overlaps
        test_box = calculate_text_bounds(text, test_x, test_y, fontsize, padding_multiplier=1.5)
        overlaps = any(boxes_overlap(test_box, occupied_box) for occupied_box in occupied_boxes)

        if not overlaps:
            return test_x, test_y

    # If all positions below overlap, fallback to the furthest position with max spacing
    return x, y - base_distance * 5.0


def calculate_dynamic_font_size(num_nodes: int, num_edges: int) -> int:
    """
    Calculate appropriate font size based on graph complexity.

    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph

    Returns:
        Font size in points
    """
    complexity = num_nodes + num_edges

    if complexity <= 10:
        return CONFIG['font_size_max']
    elif complexity <= 20:
        return CONFIG['font_size_base']
    else:
        return CONFIG['font_size_min']