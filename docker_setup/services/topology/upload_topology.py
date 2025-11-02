#!/usr/bin/env python3
"""
Upload a Mininet topology file to the topology visualizer service.

Usage:
    python upload_topology.py <path_to_topology.py> [--host HOST] [--port PORT]

Example:
    python upload_topology.py /path/to/testnet.py
    python upload_topology.py /path/to/testnet.py --host localhost --port 8001
"""

import argparse
import sys
import requests
from pathlib import Path


def upload_topology(file_path: str, host: str = 'localhost', port: int = 8001) -> dict:
    """
    Upload a topology file to the visualizer service.

    Args:
        file_path: Path to the Mininet topology Python file
        host: Service host (default: localhost)
        port: Service port (default: 8001)

    Returns:
        Response JSON from the server
    """
    # Validate file exists
    topology_file = Path(file_path)
    if not topology_file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not topology_file.suffix == '.py':
        raise ValueError("File must be a Python (.py) file")

    # Prepare the upload
    url = f"http://{host}:{port}/upload"

    print(f"Uploading {topology_file.name} to {url}...")

    with open(topology_file, 'rb') as f:
        files = {'file': (topology_file.name, f, 'text/x-python')}
        response = requests.post(url, files=files)

    # Check response
    if response.status_code == 200:
        data = response.json()
        print("\n✓ Upload successful!")
        print(f"  Topology ID: {data.get('topology_id')}")
        print(f"  Nodes: {data.get('nodes')}")
        print(f"  Edges: {data.get('edges')}")
        print(f"  Message: {data.get('message')}")
        print(f"\nView topology image at:")
        print(f"  http://{host}:{port}/topologyImage")
        print(f"  http://{host}:{port}/topologyImage?topology_id={data.get('topology_id')}")
        return data
    else:
        print(f"\n✗ Upload failed with status code {response.status_code}")
        try:
            error_data = response.json()
            print(f"  Error: {error_data.get('error', 'Unknown error')}")
        except:
            print(f"  Response: {response.text}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Upload a Mininet topology file to the topology visualizer service'
    )
    parser.add_argument(
        'file',
        help='Path to the Mininet topology Python file'
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help='Service host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Service port (default: 8001)'
    )

    args = parser.parse_args()

    try:
        upload_topology(args.file, args.host, args.port)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
