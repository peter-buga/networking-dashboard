"""
Topology visualization module for creating network topology images using NetworkX and matplotlib.
"""
import io
import math
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

from .config import TOPOLOGY_CONFIG as CONFIG, IMAGE_CONFIG
from .topology_utils import (
    calculate_text_bounds,
    get_edge_midpoint_and_rotation,
    find_optimal_ip_position,
    calculate_dynamic_font_size
)


class TopologyVisualizer:
    """Visualizer for creating network topology images."""

    @staticmethod
    def create_topology_image(hosts: Set[str], links: List[Tuple[str, str, str, str]],
                            ips: Dict[str, Dict[str, List[str]]], width: int = 1400, height: int = 500) -> bytes:
        """
        Create a network topology image from parsed topology data.

        Args:
            hosts: Set of host names
            links: List of (source, target, interface1, interface2) tuples
            ips: Dictionary mapping host -> interface -> [ip_addresses]
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            PNG image data as bytes
        """
        # Create NetworkX MultiGraph to handle multiple edges between nodes
        G = nx.MultiGraph()
        G.add_nodes_from(hosts)

        # Add edges with interface information
        for source, target, intf1, intf2 in links:
            G.add_edge(source, target, source_intf=intf1, target_intf=intf2)

        # Calculate dynamic font size based on graph complexity
        font_size = calculate_dynamic_font_size(len(hosts), len(links))

        # Create figure with increased margins for labels
        margin = CONFIG['figure_margin']
        fig, ax = plt.subplots(figsize=((width + margin * width)/100, (height + margin * height)/100), dpi=100)

        # Generate layout with more spacing
        pos = nx.kamada_kawai_layout(G, scale=2)

        # Draw the graph components first
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=CONFIG['node_size'], ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight='bold')

        # Track occupied text boxes for collision detection
        occupied_boxes = []

        # Add edge labels with smart positioning
        TopologyVisualizer._add_edge_labels(ax, pos, links, font_size, occupied_boxes)

        # Add IP information with smart positioning to avoid overlaps
        TopologyVisualizer._add_ip_labels(ax, pos, ips, font_size, occupied_boxes)

        # Enhanced title and styling
        ax.set_title("Network Topology", fontsize=min(16, font_size + 4), fontweight='bold', pad=20)
        ax.axis('off')

        # Adjust layout to fit all elements
        plt.tight_layout(pad=2.0)

        # Save to BytesIO buffer with configured quality
        buffer = io.BytesIO()
        plt.savefig(buffer, format=IMAGE_CONFIG['format'], dpi=IMAGE_CONFIG['dpi'],
                    bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
        buffer.seek(0)
        plt.close(fig)

        return buffer.getvalue()

    @staticmethod
    def _add_edge_labels(ax, pos: Dict[str, Tuple[float, float]], links: List[Tuple[str, str, str, str]],
                        font_size: int, occupied_boxes: List[Tuple[float, float, float, float]]):
        """
        Add edge labels with smart positioning to avoid overlaps.

        Args:
            ax: Matplotlib axis object
            pos: Dictionary mapping node names to (x, y) coordinates
            links: List of link tuples
            font_size: Base font size
            occupied_boxes: List to track occupied text areas
        """
        for source, target, intf1, intf2 in links:
            if intf1 and intf2:  # Only add label if both interfaces are defined
                label_text = f"{intf1} ↔ {intf2}"
                mid_x, mid_y, rotation = get_edge_midpoint_and_rotation(pos, source, target)

                # Offset the label slightly to avoid overlapping the edge line
                offset_x = mid_x + CONFIG['edge_label_offset'] * math.sin(math.radians(rotation))
                offset_y = mid_y - CONFIG['edge_label_offset'] * math.cos(math.radians(rotation))

                # Add text with improved styling
                ax.text(offset_x, offset_y, label_text,
                       rotation=rotation,
                       ha='center', va='center',
                       fontsize=max(font_size - 1, CONFIG['font_size_min']),
                       bbox=dict(boxstyle=f"round,pad={CONFIG['text_padding']}",
                               facecolor='white', edgecolor='gray', alpha=0.9),
                       zorder=5)

                # Track this text box with standard padding for edge labels
                text_box = calculate_text_bounds(label_text, offset_x, offset_y, font_size - 1, padding_multiplier=1.0)
                occupied_boxes.append(text_box)

    @staticmethod
    def _add_ip_labels(ax, pos: Dict[str, Tuple[float, float]], ips: Dict[str, Dict[str, List[str]]],
                      font_size: int, occupied_boxes: List[Tuple[float, float, float, float]]):
        """
        Add IP address labels with smart positioning to avoid overlaps.

        Args:
            ax: Matplotlib axis object
            pos: Dictionary mapping node names to (x, y) coordinates
            ips: Dictionary mapping host -> interface -> [ip_addresses]
            font_size: Base font size
            occupied_boxes: List to track occupied text areas
        """
        for host, interfaces in ips.items():
            if host in pos:
                ip_text = []
                for iface, ip_list in interfaces.items():
                    for ip in ip_list:
                        ip_text.append(f"{iface}: {ip}")

                if ip_text:
                    label_text = '\n'.join(ip_text)

                    # Find optimal position for IP label
                    ip_x, ip_y = find_optimal_ip_position(pos, host, occupied_boxes, label_text, font_size)

                    # Add IP text with enhanced styling
                    ax.text(ip_x, ip_y, label_text,
                           ha='center', va='center',
                           fontsize=max(font_size - 2, CONFIG['font_size_min']),
                           bbox=dict(boxstyle=f"round,pad={CONFIG['text_padding']}",
                                   facecolor='lightyellow', edgecolor='orange', alpha=0.9),
                           zorder=6)

                    # Track this text box with enhanced padding for IP labels
                    ip_box = calculate_text_bounds(label_text, ip_x, ip_y, font_size - 2, padding_multiplier=1.5)
                    occupied_boxes.append(ip_box)