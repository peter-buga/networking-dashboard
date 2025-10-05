import os
import random
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST


METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
INTERVAL_SECONDS = float(os.getenv("INTERVAL_SECONDS", "1"))


registry = CollectorRegistry()

req_counter = Counter("nd_requests_total", "Synthetic request count", ["method", "code"], registry=registry)
temp_gauge = Gauge("nd_temperature_celsius", "Synthetic temperature reading", ["zone"], registry=registry)
latency_hist = Histogram(
    "nd_request_latency_seconds",
    "Synthetic request latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=registry,
)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            data = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")


def start_metrics_http():
    server = HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
    server.serve_forever()


def loop():
    zones = ["a", "b", "c"]
    methods = ["GET", "POST", "PUT"]
    codes = ["200", "200", "200", "500", "404"]
    while True:
        # Generate values
        zone = random.choice(zones)
        method = random.choice(methods)
        code = random.choice(codes)
        temp = 18 + random.random() * 10
        latency = abs(random.gauss(0.1, 0.05))

        req_counter.labels(method=method, code=code).inc()
        temp_gauge.labels(zone=zone).set(temp)
        latency_hist.observe(latency)

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    threading.Thread(target=start_metrics_http, daemon=True).start()
    loop()
