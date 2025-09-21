#!/usr/bin/env python3
"""
Mininet Topology Image Generator Service

This service parses Mininet Python files to extract network topology information
and generates network diagram images for visualization in Grafana.
"""

import os
import re
import json
import time
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont


class MininetTopologyImageGenerator:
    """Generator for creating network topology images from Mininet Python files."""
    
    def __init__(self):
        self.hosts = {}
        self.links = []
        self.ip_addresses = {}
        
        # Color scheme for different node types
        self.node_colors = {
            'host': '#58a6ff',      # Blue
            'edge': '#f85149',      # Red  
            'core': '#56d364',      # Green
            'switch': '#ffa657',    # Orange
            'unknown': '#8b949e'    # Gray
        }
        
    def parse_file(self, file_path):
        """Parse a Mininet Python file and extract topology information."""
        self.hosts = {}
        self.links = []
        self.ip_addresses = {}
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract hosts, links, and IP addresses
        self._extract_hosts(content)
        self._extract_links(content)
        self._extract_ip_addresses(content)
        
        # Create nodes from links if hosts dict is empty
        self._ensure_all_nodes_exist()
        
        return True
    
    def _extract_hosts(self, content):
        """Extract host definitions from addHost calls."""
        # Pattern to match direct net.addHost calls
        host_pattern = r'net\.addHost\(["\'](\w+)["\'](?:,\s*ip=([^)]+))?\)'
        matches = re.findall(host_pattern, content)
        
        for hostname, ip in matches:
            self.hosts[hostname] = {
                'name': hostname,
                'type': 'host',
                'ip': ip.strip().strip('"\'') if ip else None
            }
        
        # Also look for host lists - pattern like: hosts = ["host1", "edge1", ...]
        host_list_pattern = r'hosts\s*=\s*\[((?:["\'][^"\']*["\'],?\s*)+)\]'
        list_matches = re.findall(host_list_pattern, content)
        
        for host_list_str in list_matches:
            # Extract individual hostnames from the list
            hostname_pattern = r'["\']([^"\']+)["\']'
            hostnames = re.findall(hostname_pattern, host_list_str)
            
            for hostname in hostnames:
                if hostname not in self.hosts:
                    self.hosts[hostname] = {
                        'name': hostname,
                        'type': self._determine_node_type(hostname),
                        'ip': None
                    }
    
    def _extract_links(self, content):
        """Extract link definitions from addLink calls."""
        # Pattern to match net.addLink calls with variables (not strings)
        link_pattern = r'net\.addLink\((\w+),\s*(\w+)(?:,\s*intfName1=["\']([^"\']+)["\'])?(?:,\s*intfName2=["\']([^"\']+)["\'])?\)'
        matches = re.findall(link_pattern, content)
        
        for host1, host2, intf1, intf2 in matches:
            link = {
                'source': host1,
                'target': host2,
                'source_interface': intf1 if intf1 else None,
                'target_interface': intf2 if intf2 else None
            }
            self.links.append(link)
    
    def _extract_ip_addresses(self, content):
        """Extract IP addresses from 'ip a a' commands."""
        # Pattern to match ip address assignment commands
        ip_pattern = r'(\w+)\.cmd\(["\']ip\s+a\s+a\s+([0-9.]+/\d+)\s+dev\s+(\w+)["\'][^)]*\)'
        matches = re.findall(ip_pattern, content)
        
        for hostname, ip_cidr, interface in matches:
            if hostname not in self.ip_addresses:
                self.ip_addresses[hostname] = {}
            if interface not in self.ip_addresses[hostname]:
                self.ip_addresses[hostname][interface] = []
            self.ip_addresses[hostname][interface].append(ip_cidr)
    
    def _ensure_all_nodes_exist(self):
        """Ensure all hosts referenced in links exist in hosts dict."""
        all_hostnames = set()
        for link in self.links:
            all_hostnames.add(link['source'])
            all_hostnames.add(link['target'])
        
        for hostname in all_hostnames:
            if hostname not in self.hosts:
                self.hosts[hostname] = {
                    'name': hostname,
                    'type': self._determine_node_type(hostname),
                    'ip': None
                }
    
    def _determine_node_type(self, hostname):
        """Determine the type of node based on naming convention."""
        hostname_lower = hostname.lower()
        if 'host' in hostname_lower:
            return 'host'
        elif 'edge' in hostname_lower:
            return 'edge'
        elif 'core' in hostname_lower:
            return 'core'
        elif 'switch' in hostname_lower:
            return 'switch'
        else:
            return 'unknown'
    
    def generate_topology_image(self, width=1200, height=800):
        """Generate a network topology image using NetworkX and Matplotlib."""
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for hostname, host_info in self.hosts.items():
            G.add_node(hostname, 
                      node_type=host_info['type'],
                      interfaces=self.ip_addresses.get(hostname, {}))
        
        # Add edges
        for link in self.links:
            G.add_edge(link['source'], link['target'],
                      source_interface=link['source_interface'],
                      target_interface=link['target_interface'])
        
        # Create the plot
        plt.figure(figsize=(width/100, height/100))
        plt.clf()
        
        # Use spring layout for nice positioning
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes by type
        for node_type, color in self.node_colors.items():
            nodes_of_type = [node for node in G.nodes() 
                           if self.hosts[node]['type'] == node_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                                     node_color=color, node_size=2000,
                                     alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='#666666', width=2, alpha=0.6)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold',
                               font_color='white')
        
        # Add interface and IP information
        self._add_interface_labels(G, pos)
        
        # Create legend
        legend_elements = []
        for node_type, color in self.node_colors.items():
            if any(self.hosts[node]['type'] == node_type for node in G.nodes()):
                legend_elements.append(mpatches.Patch(color=color, label=node_type.title()))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Set title
        plt.title('Mininet Network Topology', fontsize=16, fontweight='bold', pad=20)
        
        # Remove axes
        plt.axis('off')
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=100, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        
        plt.close()  # Clean up to prevent memory leaks
        
        return img_buffer.getvalue()
    
    def _add_interface_labels(self, G, pos):
        """Add interface and IP address labels to edges."""
        for edge in G.edges():
            source, target = edge
            edge_data = G.edges[edge]
            
            # Calculate midpoint of edge
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Create label text
            label_parts = []
            
            if edge_data.get('source_interface'):
                source_ips = self._get_interface_ips(source, edge_data['source_interface'])
                if source_ips:
                    label_parts.append(f"{edge_data['source_interface']}: {source_ips[0]}")
                else:
                    label_parts.append(edge_data['source_interface'])
            
            if edge_data.get('target_interface'):
                target_ips = self._get_interface_ips(target, edge_data['target_interface'])
                if target_ips:
                    label_parts.append(f"{edge_data['target_interface']}: {target_ips[0]}")
                else:
                    label_parts.append(edge_data['target_interface'])
            
            if label_parts:
                label_text = ' ↔ '.join(label_parts)
                plt.text(mid_x, mid_y, label_text, fontsize=8, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _get_interface_ips(self, hostname, interface):
        """Get IP addresses for a specific interface of a host."""
        if hostname in self.ip_addresses and interface in self.ip_addresses[hostname]:
            return self.ip_addresses[hostname][interface]
        return []


class TopologyAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for topology API."""
    
    def __init__(self, *args, **kwargs):
        self.generator = MininetTopologyImageGenerator()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == '/topologyImage':
            self._handle_image_request(parsed_url)
        elif parsed_url.path == '/topologyJson':
            self._handle_topology_request(parsed_url)
        elif parsed_url.path == '/health':
            self._handle_health_request()
        else:
            self._send_error(404, "Not Found")
    
    def _handle_image_request(self, parsed_url):
        """Handle topology image generation request."""
        try:
            # Get file parameter from query string
            query_params = parse_qs(parsed_url.query)
            file_param = query_params.get('file', [])
            width = int(query_params.get('width', ['1200'])[0])
            height = int(query_params.get('height', ['800'])[0])
            
            if not file_param:
                # Default to the testnet.py file
                file_path = '/app/base_files/testnet.py'
            else:
                file_path = file_param[0]
            
            # Check if file exists
            if not os.path.exists(file_path):
                self._send_error(404, f"File not found: {file_path}")
                return
            
            # Parse the file and generate image
            self.generator.parse_file(file_path)
            image_data = self.generator.generate_topology_image(width, height)
            
            # Send image response
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(image_data)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            self.wfile.write(image_data)
            
        except Exception as e:
            print(f"Error generating topology image: {str(e)}")
            self._send_error(500, f"Error generating topology image: {str(e)}")
    
    def _handle_topology_request(self, parsed_url):
        """Handle topology JSON request (for backward compatibility)."""
        try:
            # Get file parameter from query string
            query_params = parse_qs(parsed_url.query)
            file_param = query_params.get('file', [])
            
            if not file_param:
                # Default to the testnet.py file
                file_path = '/app/base_files/testnet.py'
            else:
                file_path = file_param[0]
            
            # Check if file exists
            if not os.path.exists(file_path):
                self._send_error(404, f"File not found: {file_path}")
                return
            
            # Parse the file
            self.generator.parse_file(file_path)
            
            # Build JSON response for backward compatibility
            topology_data = {
                'nodes': [{'id': hostname, 'name': hostname, 'type': info['type'], 
                          'interfaces': self.generator.ip_addresses.get(hostname, {})}
                         for hostname, info in self.generator.hosts.items()],
                'edges': [{'source': link['source'], 'target': link['target'],
                          'source_interface': link['source_interface'],
                          'target_interface': link['target_interface']}
                         for link in self.generator.links],
                'metadata': {
                    'total_hosts': len(self.generator.hosts),
                    'total_links': len(self.generator.links),
                    'parsed_at': time.time()
                }
            }
            
            # Send response
            self._send_json_response(topology_data)
            
        except Exception as e:
            print(f"Error parsing topology: {str(e)}")
            self._send_error(500, f"Error parsing topology: {str(e)}")
    
    def _handle_health_request(self):
        """Handle health check request."""
        health_data = {
            'status': 'healthy',
            'service': 'topology-parser',
            'timestamp': time.time()
        }
        self._send_json_response(health_data)
    
    def _send_json_response(self, data):
        """Send JSON response."""
        response_data = json.dumps(data, indent=2)
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()
        
        self.wfile.write(response_data.encode('utf-8'))
    
    def _send_error(self, status_code, message):
        """Send error response."""
        error_data = {
            'error': message,
            'status_code': status_code,
            'timestamp': time.time()
        }
        response_data = json.dumps(error_data, indent=2)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()
        
        self.wfile.write(response_data.encode('utf-8'))


def main():
    """Main function to start the topology service."""
    port = int(os.getenv('TOPOLOGY_PORT', '8001'))
    
    print(f"Starting Topology Parser Service on port {port}")
    
    server = HTTPServer(('0.0.0.0', port), TopologyAPIHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.shutdown()


if __name__ == '__main__':
    main()