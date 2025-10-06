# Network Notification Message Schema

## Overview
This document describes the structure of network notification messages sent by edge devices in the networking dashboard system.

## Root Level Structure

| Field | Type | Description |
|-------|------|-------------|
| `notif_seq` | number | Sequential notification number for message ordering |
| `notif_hostname` | string | Identifier of the edge device sending the notification |
| `notif_tstamp` | number | Unix timestamp with microsecond precision |
| `notif_msg` | object | Container for all device metrics and status information |

## Message Types (`notif_msg` contents)

### MIP (Message Interface Point) Objects
Used for compound edge receivers and transmitters.

**Structure:**
- `level`: number - Debug/logging level (typically 3)
- `name`: string - Component identifier
- `recv`: number - Packets received count
- `send`: number - Packets sent count
- `stream_name`: string - Associated stream identifier
- `type`: "MIP" - Message interface point type
- `object`: object - Detailed component configuration

#### MIP Object Types

**Sequence Recovery (`seqrec`)**
```json
{
  "discarded_packets": 0,
  "history_length": 512,
  "latent_error_paths": 2,
  "latent_error_resets": 0,
  "latent_errors": 0,
  "name": "pef",
  "passed_packets": 0,
  "recovery_algorithm": "vector",
  "recovery_seq_num": 65535,
  "reset_msec": 2000,
  "seq_recovery_resets": 1,
  "type": "seqrec",
  "use_init_flag": false,
  "use_reset_flag": false
}
```

**Replication (`replicate`)**
```json
{
  "name": "prf",
  "octets_passed": 2124,
  "packets_passed": 12,
  "pipelines": [
    {
      "action_count": 4,
      "mask_state": "unmasked|masked",
      "name": "tx1|tx2"
    }
  ],
  "type": "replicate"
}
```

### Interface Notifications
Physical network interface statistics.

**Pattern:** `ifNotify_<interface_name>`

| Field | Type | Description |
|-------|------|-------------|
| `recv_octets` | number | Bytes received on interface |
| `recv_packets` | number | Packets received on interface |
| `send_octets` | number | Bytes sent on interface |
| `send_packets` | number | Packets sent on interface |

### Network Node Interface (NNI)
Inter-node communication interfaces.

**Patterns:**
- `nni<N>_in` - Incoming NNI interface
- `nni<N>_out` - Outgoing NNI interface  
- `nni<N>_in parser` - Parser statistics for incoming traffic

**NNI Interface Fields:**
- `recv_octets`, `recv_packets` - Received traffic
- `send_octets`, `send_packets` - Sent traffic

**NNI Parser Fields:**
- `no match octets`, `no match packets` - Unmatched traffic
- `stream_<name> octets`, `stream_<name> packets` - Stream-specific traffic

### User Network Interface (UNI)
Customer-facing interfaces.

**Patterns:**
- `uni_rx` - UNI receiver
- `uni_tx` - UNI transmitter
- `uni_rx parser` - UNI parser statistics

### Path Components
Redundant path handling for high availability.

**Patterns:**
- `path<N>_<hostname>_rx` - Path receiver
- `path<N>_<hostname>_tx` - Path transmitter

### Standalone Components

**Generator (`gen`)**
```json
{
  "name": "gen",
  "type": "seqgen",
  "use_init_flag": false,
  "use_reset_flag": false
}
```

**Packet Elimination Function (`pef`)**
- Handles packet deduplication and error recovery
- Same structure as seqrec object type

**Packet Replication Function (`prf`)**
- Handles packet duplication across multiple paths
- Same structure as replicate object type

## Common Field Definitions

### Counters
- `*_packets`: Packet count (integer)
- `*_octets`: Byte count (integer)
- `recv`: Received packet count
- `send`: Sent packet count

### Configuration
- `level`: Debug/logging level
- `name`: Component identifier
- `type`: Component type ("MIP", "seqrec", "replicate", "seqgen")
- `stream_name`: Associated data stream

### Recovery Parameters
- `history_length`: Buffer size for packet history
- `recovery_algorithm`: Algorithm type ("vector")
- `recovery_seq_num`: Current sequence number
- `reset_msec`: Reset timeout in milliseconds
- `latent_error_*`: Error tracking counters

### Pipeline Configuration
- `action_count`: Number of actions performed
- `mask_state`: Pipeline state ("masked", "unmasked")
- `use_init_flag`, `use_reset_flag`: Control flags

## Example Usage
See `notification_example.json` for a complete example notification message from edge device "edge1".