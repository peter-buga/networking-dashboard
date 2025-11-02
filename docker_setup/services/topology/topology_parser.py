"""
Mininet Topology Parser

Parses Mininet Python files to extract network topology information:
- Hosts/nodes
- Links between nodes
- IP address assignments
"""

import re
from typing import List, Tuple, Dict


def parse_mininet_file(file_path: str) -> Tuple[List[str], List[Tuple[str, str, str, str]], Dict[str, Dict[str, List[str]]]]:
    """
    Parse a Mininet Python file to extract topology information.

    Args:
        file_path: Path to the Mininet Python file

    Returns:
        Tuple of (hosts, links, ips):
        - hosts: List of host names
        - links: List of tuples (src, dst, intf1, intf2)
        - ips: Dict mapping {hostname: {interface: [ip_addresses]}}
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract hosts
    hosts = []
    # Pattern: net.addHost('hostname', ...) or net.addHost("hostname", ...)
    host_pattern = r'net\.addHost\([\'"](\w+)[\'"]'
    hosts = re.findall(host_pattern, content)

    # Also extract hosts from the list assignment pattern like: host1, edge1, core = [net.get(n) for n in hosts]
    # First find the hosts list definition
    hosts_list_pattern = r'hosts\s*=\s*\[(.*?)\]'
    hosts_list_match = re.search(hosts_list_pattern, content, re.DOTALL)
    if hosts_list_match:
        # Extract quoted strings from the list
        hosts_from_list = re.findall(r'[\'"](\w+)[\'"]', hosts_list_match.group(1))
        # Merge with addHost results, removing duplicates
        hosts = list(set(hosts + hosts_from_list))

    # Extract links
    links = []
    # Pattern: net.addLink(host1, host2, intfName1='intf1', intfName2='intf2')
    # More flexible pattern to handle various formats
    link_pattern = r'net\.addLink\((\w+),\s*(\w+)(?:,\s*intfName1=[\'"]([^\'"]+)[\'"])?(?:,\s*intfName2=[\'"]([^\'"]+)[\'"])?\)'
    link_matches = re.findall(link_pattern, content)

    for match in link_matches:
        src, dst, intf1, intf2 = match
        # intf1 and intf2 might be empty strings if not specified
        links.append((src, dst, intf1 or '', intf2 or ''))

    # Extract IP addresses
    ips = {}
    # Pattern: hostname.cmd("ip a a 192.168.1.1/24 dev eth0")
    # Also handle: hostname.cmd('ip a a ...')
    ip_pattern = r'(\w+)\.cmd\([\'"]ip\s+a\s+a\s+([0-9.]+/\d+)\s+dev\s+(\w+)[\'"]'
    ip_matches = re.findall(ip_pattern, content)

    for hostname, ip_addr, interface in ip_matches:
        if hostname not in ips:
            ips[hostname] = {}
        if interface not in ips[hostname]:
            ips[hostname][interface] = []
        ips[hostname][interface].append(ip_addr)

    return hosts, links, ips


def format_node_label(hostname: str, ips: Dict[str, Dict[str, List[str]]]) -> str:
    """
    Format a node label with hostname and IP addresses.

    Args:
        hostname: Name of the host
        ips: IP address dictionary

    Returns:
        Formatted label string
    """
    label_lines = [hostname]

    if hostname in ips:
        for interface in sorted(ips[hostname].keys()):
            for ip_addr in ips[hostname][interface]:
                label_lines.append(f"{interface}: {ip_addr}")

    return "\n".join(label_lines)


if __name__ == "__main__":
    # Test the parser
    import sys

    if len(sys.argv) != 2:
        print("Usage: python topology_parser.py <mininet_file.py>")
        sys.exit(1)

    file_path = sys.argv[1]
    hosts, links, ips = parse_mininet_file(file_path)

    print(f"Hosts ({len(hosts)}): {hosts}")
    print(f"\nLinks ({len(links)}):")
    for src, dst, intf1, intf2 in links:
        print(f"  {src} <-> {dst} ({intf1} — {intf2})")

    print(f"\nIP Addresses:")
    for host, interfaces in ips.items():
        print(f"  {host}:")
        for intf, addrs in interfaces.items():
            print(f"    {intf}: {', '.join(addrs)}")
