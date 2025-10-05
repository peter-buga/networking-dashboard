"""
Topology parsing module for extracting network topology information from Mininet files.
"""
import re
from typing import Dict, List, Set, Tuple


# Regex patterns for topology extraction
HOST_REGEX = re.compile(r"(?:self|net)\.addHost\(['\"](\w+)['\"]\)")
HOSTS_LIST_REGEX = re.compile(r"hosts\s*=\s*\[((?:['\"][^'\"]+['\"],?\s*)+)\]")
HOST_IN_LIST_REGEX = re.compile(r"['\"]([^'\"]+)['\"]")
LINK_REGEX = re.compile(
    r"net\.addLink\((\w+),\s*(\w+)"
    r"(?:,\s*intfName1=['\"]([^'\"]+)['\"])?"
    r"(?:,\s*intfName2=['\"]([^'\"]+)['\"])?"
    r"\)"
)
IP_REGEX = re.compile(r"(\w+)\.cmd\(['\"]ip\s+a\s+a\s+(\d+\.\d+\.\d+\.\d+/\d+)\s+dev\s+(\w+)['\"]\)")


class TopologyParser:
    """Parser for extracting network topology from Mininet Python files."""

    @staticmethod
    def parse_topology_file(file_path: str) -> Tuple[Set[str], List[Tuple[str, str, str, str]], Dict[str, Dict[str, List[str]]]]:
        """
        Parse a Mininet topology file and extract hosts, links, and IP configurations.

        Args:
            file_path: Path to the topology file

        Returns:
            Tuple of (hosts, links, ips) where:
            - hosts: Set of host names
            - links: List of (source, target, interface1, interface2) tuples
            - ips: Dict mapping host -> interface -> [ip_addresses]
        """
        hosts = set()
        links = []  # (source, target, intf1, intf2)
        ips = {}  # host -> interface -> [ip_addresses]

        with open(file_path, 'r') as f:
            for line in f:
                # Parse hosts from list declarations
                if (match := HOSTS_LIST_REGEX.search(line)):
                    inner = match.group(1)
                    for host_match in HOST_IN_LIST_REGEX.finditer(inner):
                        hosts.add(host_match.group(1))
                    continue

                # Parse individual host additions
                if (match := HOST_REGEX.search(line)):
                    hosts.add(match.group(1))
                    continue

                # Parse network links
                if (match := LINK_REGEX.search(line)):
                    source, target, intf1, intf2 = match.groups()
                    links.append((source, target, intf1, intf2))
                    continue

                # Parse IP address assignments
                if (match := IP_REGEX.search(line)):
                    host, ip, interface = match.groups()
                    ips.setdefault(host, {}).setdefault(interface, []).append(ip)

        # Ensure all nodes referenced by links exist in hosts set
        for source, target, _, _ in links:
            if source not in hosts:
                hosts.add(source)
            if target not in hosts:
                hosts.add(target)

        return hosts, links, ips

    @staticmethod
    def create_topology_json(hosts: Set[str], links: List[Tuple[str, str, str, str]],
                           ips: Dict[str, Dict[str, List[str]]], parsed_at: float) -> Dict:
        """
        Create JSON representation of the topology data.

        Args:
            hosts: Set of host names
            links: List of link tuples
            ips: IP address mappings
            parsed_at: Timestamp when parsing occurred

        Returns:
            Dictionary with topology data in JSON format
        """
        return {
            'nodes': [{'id': node} for node in sorted(hosts)],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'source_interface': intf1,
                    'target_interface': intf2,
                } for source, target, intf1, intf2 in links
            ],
            'ips': ips,
            'metadata': {
                'total_hosts': len(hosts),
                'total_links': len(links),
                'parsed_at': parsed_at,
            }
        }