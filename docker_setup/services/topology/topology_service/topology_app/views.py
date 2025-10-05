import os
import time

from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_GET

from .config import DEFAULT_TOPOLOGY_FILE, IMAGE_CONFIG
from .topology_parser import TopologyParser
from .topology_visualizer import TopologyVisualizer





@require_GET
def topology_image(request):
    """Generate topology visualization image."""
    file_path = request.GET.get('file', DEFAULT_TOPOLOGY_FILE)

    # Validate width and height parameters
    try:
        width = int(request.GET.get('width', IMAGE_CONFIG['default_width']))
        height = int(request.GET.get('height', IMAGE_CONFIG['default_height']))
        if width <= 0 or height <= 0:
            return JsonResponse({'error': 'Width and height must be positive integers.'}, status=400)
    except ValueError:
        return JsonResponse({'error': 'Invalid width or height parameter.'}, status=400)

    if not os.path.exists(file_path):
        return JsonResponse({'error': f'File not found: {file_path}'}, status=404)

    # Prevent path traversal
    if not os.path.isfile(file_path) or os.path.abspath(file_path).startswith(os.path.abspath(os.path.dirname(__file__))):
        return JsonResponse({'error': 'Invalid file path.'}, status=400)

    try:
        # Parse topology data
        hosts, links, ips = TopologyParser.parse_topology_file(file_path)
    except Exception as e:
        return JsonResponse({'error': f'Error parsing topology file: {str(e)}'}, status=500)

    try:
        # Generate visualization
        image_data = TopologyVisualizer.create_topology_image(hosts, links, ips, width, height)
    except Exception as e:
        return JsonResponse({'error': f'Error generating topology image: {str(e)}'}, status=500)

    return HttpResponse(image_data, content_type='image/png')
   


@require_GET
def topology_json(request):
    """Get topology data as JSON."""
    file_path = request.GET.get('file', DEFAULT_TOPOLOGY_FILE)
    if not os.path.exists(file_path):
        return JsonResponse({'error': f'File not found: {file_path}'}, status=404)

    # Parse topology data
    hosts, links, ips = TopologyParser.parse_topology_file(file_path)

    # Create JSON response
    data = TopologyParser.create_topology_json(hosts, links, ips, time.time())
    return JsonResponse(data)


@require_GET
def health(_request):
    """Health check endpoint."""
    return JsonResponse({'status': 'healthy', 'service': 'topology-django', 'timestamp': time.time()})
