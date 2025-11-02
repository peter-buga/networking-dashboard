"""
Topology Visualizer Flask API

Provides endpoints to:
- Upload Mininet topology files
- Generate and serve topology visualizations
"""

import os
import uuid
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io

from topology_parser import parse_mininet_file
from topology_visualizer import generate_topology_image


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('/tmp/topologies')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {'py'}
MAX_FILE_AGE_SECONDS = 3600  # 1 hour

# In-memory state
latest_topology_id = None
topology_metadata = {}  # {topology_id: {'file_path': ..., 'timestamp': ..., 'nodes': ..., 'edges': ...}}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_files():
    """Remove topology files older than MAX_FILE_AGE_SECONDS."""
    current_time = time.time()
    to_remove = []

    for topology_id, metadata in topology_metadata.items():
        if current_time - metadata['timestamp'] > MAX_FILE_AGE_SECONDS:
            file_path = metadata['file_path']
            if os.path.exists(file_path):
                os.remove(file_path)
            to_remove.append(topology_id)

    for topology_id in to_remove:
        del topology_metadata[topology_id]


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'topology-visualizer'}), 200


@app.route('/upload', methods=['POST'])
def upload_topology():
    """
    Upload a Mininet topology Python file.

    Returns:
        JSON with topology_id and basic statistics
    """
    # Cleanup old files before processing new upload
    cleanup_old_files()

    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File must be a Python (.py) file'}), 400

    try:
        # Generate unique ID for this topology
        topology_id = str(uuid.uuid4())

        # Save file with UUID filename
        filename = f"{topology_id}.py"
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)

        # Parse the topology file
        hosts, links, ips = parse_mininet_file(str(file_path))

        # Store metadata
        global latest_topology_id
        latest_topology_id = topology_id
        topology_metadata[topology_id] = {
            'file_path': str(file_path),
            'timestamp': time.time(),
            'nodes': len(hosts),
            'edges': len(links),
            'original_filename': secure_filename(file.filename)
        }

        return jsonify({
            'status': 'success',
            'topology_id': topology_id,
            'nodes': len(hosts),
            'edges': len(links),
            'message': f'Topology uploaded successfully: {len(hosts)} nodes, {len(links)} edges'
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to process topology file: {str(e)}'}), 500


@app.route('/topologyImage', methods=['GET'])
def get_topology_image():
    """
    Generate and return topology visualization image.

    Query Parameters:
        - topology_id: UUID of uploaded topology (optional, defaults to latest)
        - width: Image width in pixels (default: 1600)
        - height: Image height in pixels (default: 900)
        - format: Image format - 'svg' or 'png' (default: svg)

    Returns:
        Image file (SVG or PNG)
    """
    # Get parameters
    topology_id = request.args.get('topology_id', latest_topology_id)
    width = int(request.args.get('width', 1600))
    height = int(request.args.get('height', 900))
    img_format = request.args.get('format', 'svg').lower()

    # Validate format
    if img_format not in ['svg', 'png']:
        return jsonify({'error': 'Format must be svg or png'}), 400

    # Check if topology exists
    if topology_id is None:
        return jsonify({'error': 'No topology uploaded yet. Use POST /upload to upload a topology file.'}), 404

    if topology_id not in topology_metadata:
        return jsonify({'error': f'Topology {topology_id} not found'}), 404

    try:
        # Get file path
        file_path = topology_metadata[topology_id]['file_path']

        if not os.path.exists(file_path):
            return jsonify({'error': 'Topology file no longer exists'}), 404

        # Parse topology
        hosts, links, ips = parse_mininet_file(file_path)

        # Generate visualization
        image_bytes = generate_topology_image(hosts, links, ips, width, height, img_format)

        # Determine content type
        content_type = 'image/svg+xml' if img_format == 'svg' else 'image/png'

        # Return image
        return send_file(
            io.BytesIO(image_bytes),
            mimetype=content_type,
            as_attachment=False,
            download_name=f'topology.{img_format}'
        )

    except Exception as e:
        return jsonify({'error': f'Failed to generate topology image: {str(e)}'}), 500


@app.route('/topologies', methods=['GET'])
def list_topologies():
    """
    List all available topology IDs with metadata.

    Returns:
        JSON list of topologies
    """
    cleanup_old_files()

    topologies = []
    for topology_id, metadata in topology_metadata.items():
        topologies.append({
            'topology_id': topology_id,
            'nodes': metadata['nodes'],
            'edges': metadata['edges'],
            'original_filename': metadata['original_filename'],
            'uploaded_at': metadata['timestamp'],
            'is_latest': topology_id == latest_topology_id
        })

    return jsonify({
        'topologies': topologies,
        'count': len(topologies)
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('TOPOLOGY_PORT', 8001))
    app.run(host='0.0.0.0', port=port, debug=False)
