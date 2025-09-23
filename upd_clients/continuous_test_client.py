#!/usr/bin/env python3
"""
Enhanced test script to generate continuous UDP notifications with varying data.
"""

import json
import socket
import time
import random
import threading
from datetime import datetime

def create_transaction_notification(hostname, seq):
    """Create a transaction notification with random values."""
    return {
        "notif_hostname": hostname,
        "notif_msg": {
            "push_level": "INFO",
            "transaction": {
                "add_ifaces": random.randint(1, 20),
                "add_objects": random.randint(1, 10),
                "add_streams": random.randint(1, 5),
                "committed": f"{hostname}.ini"
            }
        },
        "notif_seq": seq,
        "notif_tstamp": time.time()
    }

def create_r2dtwo_notification(hostname, seq):
    """Create an r2dtwo status notification."""
    statuses = ["startup completed", "operational", "maintenance mode", "shutting down"]
    return {
        "notif_hostname": hostname,
        "notif_msg": {
            "push_level": "INFO",
            "r2dtwo": {
                "status": random.choice(statuses)
            }
        },
        "notif_seq": seq,
        "notif_tstamp": time.time()
    }

def create_triggered_source_notification(hostname, seq):
    """Create a triggered source notification with MEP data."""
    mep_names = [f"c_{hostname}_tx", f"path1_{hostname}_tx", f"path2_{hostname}_tx"]
    
    # Create MEP entries
    mep_entries = []
    for mep_name in mep_names:
        mep_entries.append({
            "mask_signal_state": random.choice(["masked", "unmasked"]),
            "name": mep_name,
            "oam_octets_passed": random.randint(0, 500000),
            "oam_packets_passed": random.randint(0, 2000),
            "octets_passed": random.randint(0, 1000),
            "packets_passed": random.randint(0, 50),
            "type": "mep_state"
        })
    
    # Add PRF entry
    mep_entries.append({
        "name": "prf",
        "octets_passed": random.randint(100000, 1000000),
        "packets_passed": random.randint(500, 5000),
        "pipelines": [
            {
                "action_count": random.randint(1, 10),
                "mask_state": random.choice(["masked", "unmasked"]),
                "name": "tx1"
            },
            {
                "action_count": random.randint(1, 10),
                "mask_state": random.choice(["masked", "unmasked"]),
                "name": "tx2"
            }
        ],
        "type": "replicate"
    })
    
    targets = ["c_edge2_rx", "c_edge3_rx", "c_edge4_rx"]
    streams = ["stream_uni", "stream_bi", "stream_multi"]
    
    return {
        "notif_hostname": hostname,
        "notif_msg": {
            "push_level": "INFO",
            "triggered_source": {
                "level": random.randint(1, 5),
                "mep": mep_entries,
                "node_id": random.randint(1, 10),
                "seq": random.randint(0, 1000),
                "session": random.randint(1, 10),
                "source": f"c_{hostname}_tx",
                "stream": random.choice(streams),
                "target": random.choice(targets)
            }
        },
        "notif_seq": seq,
        "notif_tstamp": time.time()
    }

def send_notification(sock, notification, target_host="localhost", target_port=6000):
    """Send a single notification via UDP."""
    try:
        data = json.dumps(notification).encode('utf-8')
        sock.sendto(data, (target_host, target_port))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent {notification['notif_msg'].keys()} from {notification['notif_hostname']}")
    except Exception as e:
        print(f"Error sending notification: {e}")

def continuous_sender(hostname, sock, stop_event):
    """Continuously send notifications from a specific hostname."""
    seq = 0
    notification_types = [
        ("transaction", create_transaction_notification),
        ("r2dtwo", create_r2dtwo_notification), 
        ("triggered_source", create_triggered_source_notification)
    ]
    
    while not stop_event.is_set():
        # Choose notification type with different probabilities
        # triggered_source is most common (60%), transaction (30%), r2dtwo (10%)
        rand = random.random()
        if rand < 0.6:
            notification = create_triggered_source_notification(hostname, seq)
        elif rand < 0.9:
            notification = create_transaction_notification(hostname, seq)
        else:
            notification = create_r2dtwo_notification(hostname, seq)
        
        send_notification(sock, notification)
        seq += 1
        
        # Random interval between 2-8 seconds
        time.sleep(random.uniform(2, 8))

def main():
    print("Starting continuous network node simulator...")
    print("Press Ctrl+C to stop")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Node hostnames
    hostnames = ["edge1", "edge2", "edge3", "edge4"]
    
    # Create stop event for graceful shutdown
    stop_event = threading.Event()
    
    # Start sender threads for each hostname
    threads = []
    for hostname in hostnames:
        thread = threading.Thread(target=continuous_sender, args=(hostname, sock, stop_event))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        print(f"Started sender thread for {hostname}")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
        
        # Wait for threads to finish (with timeout)
        for thread in threads:
            thread.join(timeout=2)
        
        sock.close()
        print("Stopped all senders")

if __name__ == "__main__":
    main()