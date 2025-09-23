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
    width = int(request.GET.get('width', IMAGE_CONFIG['default_width']))
    height = int(request.GET.get('height', IMAGE_CONFIG['default_height']))

    if not os.path.exists(file_path):
        return JsonResponse({'error': f'File not found: {file_path}'}, status=404)

    # Parse topology data
    hosts, links, ips = TopologyParser.parse_topology_file(file_path)

    # Generate visualization
    image_data = TopologyVisualizer.create_topology_image(hosts, links, ips, width, height)

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
