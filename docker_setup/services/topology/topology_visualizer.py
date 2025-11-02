"""
Topology Visualizer using Graphviz

Generates network topology diagrams from parsed Mininet data.
Uses pygraphviz for professional network topology visualization.
"""

import pygraphviz as pgv
from typing import List, Tuple, Dict


def generate_topology_image(
    hosts: List[str],
    links: List[Tuple[str, str, str, str]],
    ips: Dict[str, Dict[str, List[str]]],
    width: int = 1600,
    height: int = 900,
    format: str = 'svg'
) -> bytes:
    """
    Generate a network topology visualization using Graphviz.

    Args:
        hosts: List of host names
        links: List of link tuples (src, dst, intf1, intf2)
        ips: IP address dictionary {hostname: {interface: [ips]}}
        width: Image width in pixels
        height: Image height in pixels
        format: Output format ('svg' or 'png')

    Returns:
        Image bytes in specified format
    """
    # Create a directed graph
    graph = pgv.AGraph(strict=False, directed=False)

    # Configure graph attributes
    dpi = 72  # Use 72 DPI to match Graphviz internal calculations
    width_inches = width / dpi
    height_inches = height / dpi

    graph.graph_attr.update(
        dpi=str(dpi),
        bgcolor='white',
        rankdir='LR',  # Left to right layout for better 16:9 aspect ratio
        nodesep='1.0',
        ranksep='1.5',
        margin='0.1',
        size=f"{width_inches},{height_inches}!"
    )

    # Default node attributes
    graph.node_attr.update(
        shape='box',
        style='filled,rounded',
        fontname='Arial',
        fontsize='8',
        margin='0.15,0.1'
    )

    # Default edge attributes
    graph.edge_attr.update(
        color='#666666',
        fontname='Arial',
        fontsize='7',
        fontcolor='#555555'
    )

    # Calculate node degrees for styling
    degrees = {}
    for src, dst, _, _ in links:
        degrees[src] = degrees.get(src, 0) + 1
        degrees[dst] = degrees.get(dst, 0) + 1

    # Add nodes with styling based on degree
    for host in hosts:
        degree = degrees.get(host, 0)

        # Create label with IPs
        label = _format_node_label(host, ips)

        # Color and size based on node degree
        if degree == 1:
            # Leaf nodes (endpoints)
            fillcolor = '#87CEEB'  # Light blue
            fontsize = '8'
        elif degree == 2:
            # Edge nodes
            fillcolor = '#90EE90'  # Light green
            fontsize = '9'
        else:
            # Core nodes (hubs)
            fillcolor = '#F08080'  # Light coral
            fontsize = '10'

        graph.add_node(
            host,
            label=label,
            fillcolor=fillcolor,
            fontsize=fontsize
        )

    # Add edges with interface labels
    # Graphviz automatically handles parallel edges
    for src, dst, intf1, intf2 in links:
        edge_label = _format_edge_label(intf1, intf2)

        if edge_label:
            graph.add_edge(src, dst, label=edge_label)
        else:
            graph.add_edge(src, dst)

    # Generate the visualization
    image_bytes = graph.draw(format=format, prog='dot')

    return image_bytes


def _format_node_label(hostname: str, ips: Dict[str, Dict[str, List[str]]]) -> str:
    """
    Format node label with hostname and IP addresses.

    Args:
        hostname: Name of the host
        ips: IP address dictionary

    Returns:
        Formatted label string
    """
    label_lines = [f"<b>{hostname}</b>"]

    if hostname in ips:
        for interface in sorted(ips[hostname].keys()):
            for ip_addr in ips[hostname][interface]:
                label_lines.append(f"{interface}: {ip_addr}")

    return "<" + "<br/>".join(label_lines) + ">"


def _format_edge_label(intf1: str, intf2: str) -> str:
    """
    Format edge label from interface names.

    Args:
        intf1: First interface name
        intf2: Second interface name

    Returns:
        Formatted edge label
    """
    if intf1 and intf2:
        return f"{intf1} — {intf2}"
    elif intf1:
        return intf1
    elif intf2:
        return intf2
    return ""


if __name__ == "__main__":
    # Test visualization
    import sys
    from topology_parser import parse_mininet_file

    if len(sys.argv) != 2:
        print("Usage: python topology_visualizer.py <mininet_file.py>")
        sys.exit(1)

    file_path = sys.argv[1]
    hosts, links, ips = parse_mininet_file(file_path)

    print(f"Generating topology visualization for {len(hosts)} hosts and {len(links)} links...")

    image_bytes = generate_topology_image(hosts, links, ips, format='png')

    output_file = 'topology_test.png'
    with open(output_file, 'wb') as f:
        f.write(image_bytes)

    print(f"Saved visualization to {output_file}")
