#!/usr/bin/env python3

import argparse
import ipaddress
import json
import signal
import socket
import sys
from typing import Tuple
import logging
from pathlib import Path

# Configure logging with both console and file output
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "multipart_json_udp_receiver.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from notification_receiver import NotificationReceiver
from metrics_exporter import MetricsExporter

DEFAULT_IP = "::"
DEFAULT_PORT = 5678
DEFAULT_METRICS_HOST = "0.0.0.0"
DEFAULT_METRICS_PORT = 9100


def signal_handler(sig, frame):
    """Exit cleanly when Ctrl+C is pressed."""
    logger.info("Ctrl+C pressed, exiting.")
    sys.exit(0)

def parse_cli_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Receive R2DTWO multipart JSON notifications over UDP, reassemble them, "
            "and expose metrics API for prometheus."
        )
    )
    parser.add_argument(
        "listen_address",
        nargs="?",
        default=DEFAULT_IP,
        help="IP address to bind the UDP listener to (default: :: for dual stack)",
    )
    parser.add_argument(
        "listen_port",
        nargs="?",
        type=int,
        default=DEFAULT_PORT,
        help=f"UDP port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--metrics-host",
        default=DEFAULT_METRICS_HOST,
        help="Host/IP address to bind the Prometheus metrics HTTP exporter (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=DEFAULT_METRICS_PORT,
        help=f"TCP port for the Prometheus metrics HTTP exporter (default: {DEFAULT_METRICS_PORT})",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable the Prometheus metrics exporter",
    )
    return parser.parse_args()


def validate_port(port: int) -> None:
    """Validate that the port is within the allowed range."""
    if port < 1024 or port > 65535:
        raise ValueError(f"Invalid port {port}; provide a value between 1024 and 65535")


def create_udp_socket(bind_address: str, bind_port: int) -> socket.socket:
    """Create and bind a UDP socket for the given address."""
    try:
        ip = ipaddress.ip_address(bind_address)
    except ValueError as exc:
        raise ValueError(f"Invalid IP address: {bind_address}") from exc

    family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_DGRAM)

    if family == socket.AF_INET6:
        try:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            # Not all platforms expose IPV6_V6ONLY; ignore if unavailable.
            pass

    sock.bind((bind_address, bind_port))
    return sock


def resolve_sender(addr: Tuple[str, ...]) -> Tuple[str, str]:
    """Resolve sender address to numeric host/port strings."""
    try:
        host_ip, port = socket.getnameinfo(addr, socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)
    except socket.gaierror:
        # Fallback: addr already contains the numeric representation.
        host_ip, port = addr[0], str(addr[1])
    return host_ip, port


def main():
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_cli_arguments()

    try:
        validate_port(args.listen_port)
    except ValueError as exc:
        logger.error(exc)
        sys.exit(1)

    try:
        udp_socket = create_udp_socket(args.listen_address, args.listen_port)
    except ValueError as exc:
        logger.error(exc)
        sys.exit(1)
    except OSError as exc:
        logger.error(f"Unable to bind UDP socket on {args.listen_address}:{args.listen_port} -> {exc}")
        sys.exit(1)

    metrics = MetricsExporter(args.metrics_host, args.metrics_port, args.no_metrics)

    receiver = NotificationReceiver()

    logger.info(f"JSON receiver server started on {args.listen_address} : {args.listen_port}")

    message_count = 0

    while True:
        data, addr = udp_socket.recvfrom(4096)
        
        host_ip, port = resolve_sender(addr)

        json_received = receiver.process_notification(host_ip, port, data)
        
        if json_received is None:
            continue
        
        # Update detailed notification metrics
        metrics.update_notification_metrics(json_received)
        
        logger.info("========== Notification ==========")
        logger.info(json.dumps(json_received, indent=4))
        logger.info("........... End ...........")
        logger.info(f"Message count: {message_count}")

        message_count += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            input("Press Enter to exit...")
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)
