def topology_image(request):
    file_path = request.GET.get('file', DEFAULT_FILE)
    width = int(request.GET.get('width', '1200'))
    height = int(request.GET.get('height', '800'))
    engine = request.GET.get('engine', 'dot')

    if not os.path.exists(file_path):
        return JsonResponse({'error': f'File not found: {file_path}'}, status=404)

    hosts, links, ips = parse_topology(file_path)

    dpi = 72  # Standard for SVG
    w_in = width / dpi
    h_in = height / dpi
    fontsize = max(8, width // 100)
    penwidth = max(1, width // 700)
   
    g = Graph('topology', engine=engine, format='svg')
    g.attr(dpi=str(dpi), size=f'{w_in},{h_in}!', pad='0', margin='0', rankdir='LR', bgcolor='white')
    g.node_attr.update(fontname='Helvetica', fontsize=str(fontsize), style='filled')
    g.edge_attr.update(color='#333333', penwidth=str(penwidth))

    def node_label(name: str) -> str:
        lines = [name]
        info = ips.get(name, {})
        for iface in sorted(info.keys()):
            for ip in info[iface]:
                lines.append(f"{iface}: {ip}")
        return "\n".join(lines)

    # Compute node degrees and group nodes into tiers by degree (1-edge nodes, 2-edge nodes, ...)
    degree = {n: 0 for n in hosts}
    for a, b, i1, i2 in links:
        degree.setdefault(a, 0)
        degree.setdefault(b, 0)
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    tiers = {}
    for n, d in degree.items():
        # ignore nodes with zero degree (they'll still be shown, but grouped separately)
        tiers.setdefault(d, []).append(n)

    # Create subgraphs (clusters) for each non-empty degree tier and force same-rank
    ordered_degrees = sorted(k for k in tiers.keys() if tiers.get(k))
    first_nodes = []
    for d in ordered_degrees:
        nodes = sorted(tiers[d])
        with g.subgraph(name=f'cluster_tier_{d}') as c:
            # hide cluster box by removing color and label
            c.attr(rank='same', color='none', label='', margin='0')
            for n in nodes:
                # automatic node sizing: base size scales with image width, larger degree -> larger node
                base = max(1, width / 800)  # inches, min 1in
                size_factor = 1 + (d - 1) * 0.1
                node_size_in = round(base * size_factor, 2)
                # node appearance with fixed size
                c.node(n, label=node_label(n), shape='circle', fillcolor='skyblue',
                       width=str(node_size_in), height=str(node_size_in), fixedsize='true')
        # remember first node to chain tiers and enforce ordering
        if nodes:
            first_nodes.append(nodes[0])

    # Add invisible edges between first nodes of consecutive tiers to enforce ordering
    for i in range(len(first_nodes) - 1):
        g.edge(first_nodes[i], first_nodes[i+1], style='invis')

    # Now draw real edges
    for a, b, i1, i2 in links:
        label = None
        if i1 or i2:
            left = i1 or ''
            right = i2 or ''
            label = f'{left} — {right}'.strip(' —')
        if label:
            g.edge(a, b, label=label, fontsize=str(fontsize), fontcolor='#555555')
        else:
            g.edge(a, b)

    svg_bytes = g.pipe(format='svg')
    return HttpResponse(svg_bytes, content_type='image/svg+xml')