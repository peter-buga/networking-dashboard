#!/usr/bin/env python3
"""
Test script to send sample UDP notifications to the collector service.
"""

import json
import socket
import time

# Sample notifications based on the notification_mix.txt format
sample_notifications = [
    {
        "notif_hostname": "edge1",
        "notif_msg": {
            "push_level": "INFO",
            "transaction": {
                "add_ifaces": 10,
                "add_objects": 3,
                "add_streams": 3,
                "committed": "edge1.ini"
            }
        },
        "notif_seq": 0,
        "notif_tstamp": 1744037778.721275
    },
    {
        "notif_hostname": "edge1",
        "notif_msg": {
            "push_level": "INFO",
            "r2dtwo": {
                "status": "startup completed"
            }
        },
        "notif_seq": 1,
        "notif_tstamp": 1744037778.7405703
    },
    {
        "notif_hostname": "edge1",
        "notif_msg": {
            "push_level": "INFO",
            "triggered_source": {
                "level": 3,
                "mep": [
                    {
                        "mask_signal_state": "unmasked",
                        "name": "c_edge1_tx",
                        "oam_octets_passed": 364089,
                        "oam_packets_passed": 1036,
                        "octets_passed": 460,
                        "packets_passed": 5,
                        "type": "mep_state"
                    },
                    {
                        "mask_signal_state": "unmasked",
                        "name": "path1_edge1_tx",
                        "oam_octets_passed": 0,
                        "oam_packets_passed": 0,
                        "octets_passed": 460,
                        "packets_passed": 5,
                        "type": "mep_state"
                    },
                    {
                        "name": "prf",
                        "octets_passed": 364089,
                        "packets_passed": 1036,
                        "pipelines": [
                            {
                                "action_count": 3,
                                "mask_state": "unmasked",
                                "name": "tx1"
                            },
                            {
                                "action_count": 3,
                                "mask_state": "unmasked",
                                "name": "tx2"
                            }
                        ],
                        "type": "replicate"
                    }
                ],
                "node_id": 1,
                "seq": 0,
                "session": 2,
                "source": "c_edge1_tx",
                "stream": "stream_uni",
                "target": "c_edge2_rx"
            }
        },
        "notif_seq": 1407,
        "notif_tstamp": 1744039850.6794293
    }
]

def send_notification(sock, notification, target_host="localhost", target_port=6000):
    """Send a single notification via UDP."""
    data = json.dumps(notification).encode('utf-8')
    sock.sendto(data, (target_host, target_port))
    print(f"Sent notification from {notification['notif_hostname']}, seq={notification['notif_seq']}")

def main():
    print("Starting UDP notification test client...")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Send all sample notifications
        for notification in sample_notifications:
            send_notification(sock, notification)
            time.sleep(1)  # Wait 1 second between notifications
        
        # Send a few more with varying data to test metrics
        for i in range(5):
            notification = {
                "notif_hostname": f"edge{(i % 3) + 1}",
                "notif_msg": {
                    "push_level": "INFO",
                    "transaction": {
                        "add_ifaces": i + 1,
                        "add_objects": i * 2,
                        "add_streams": i + 3,
                        "committed": f"edge{i}.ini"
                    }
                },
                "notif_seq": i + 10,
                "notif_tstamp": time.time()
            }
            send_notification(sock, notification)
            time.sleep(0.5)
        
        print(f"Successfully sent {len(sample_notifications) + 5} notifications")
        
    except Exception as e:
        print(f"Error sending notifications: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()