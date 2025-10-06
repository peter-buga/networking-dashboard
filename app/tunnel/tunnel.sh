sudo socat TCP4-LISTEN:19100,fork,reuseaddr TCP:10.200.0.1:9100
xdg-open http://localhost:19100/metrics