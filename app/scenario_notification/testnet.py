#!/usr/bin/python3
from mininet.net import Mininet
from mininet.cli import CLI
import time
from pathlib import Path

"""
      ╔═══════╗                             ╔═══════╗
      ║ host1 ║                             ║ host2 ║
  ┌───╚═══════╝───┐                      ┌──╚═══════╝───┐
  │   (talker)    │                      │  (listener)  │
  │               │                      │              │
  │ue0:192.168.2.2│                      │ue0:10.10.11.2│
  └────┬──────────┘                      └─────────┬────┘
       │                                           │
       │                                           │
┌──────┴─────────┐       ┌───────────┐             │
│    eno1        │       │           │             │
│                │       │           │       ┌─────┴───┐
│          ens2f0├───────┤ens2f0     │       │   eno2  │
│                │       │           │       │         │
│                │       │       eno1├───────┤eno1     │
│          ens2f1├───────│ens2f1     │       │         │
│                │       │           │       │  (pef)  │
│     (prf)      │       │    eno2   │       └╔═══════╗┘
└───╔═══════╗────┘       └─╔═══════╗─┘        ║ edge2 ║
    ║ edge1 ║              ║ core  ║          ╚═══════╝
    ╚═══════╝              ╚═══════╝           R2DTWO
     R2DTWO                     |
                                | 
                                |
                        ┌───────────┐
                        |  obs-eth0 |
                        |           |
                        |  obs host |
                        └───────────┘
"""


def main():
    import os
    import sys

    if len(sys.argv) == 1:
        automip = False
    elif len(sys.argv) == 2 and sys.argv[1].lower() == "automip":
        automip = True
    else:
        print("Unknown option. AutoMIP is the only accepted parameter.")
        exit(1)

    if automip:
        print("Starting AutoMIP version")
    else:
        print("Starting manual MIP version")

    os.system("killall xterm >/dev/null 2>&1")

    net = Mininet()

    hosts = ["host1", "edge1", "core", "edge2", "host2", "obs"]
    for hostname in hosts:
        net.addHost(hostname, ip=None)

    host1, edge1, core, edge2, host2, obs = [net.get(n) for n in hosts]

    # Root-namespace bridge host to expose observability metrics outside Mininet
    obsbridge = net.addHost("obsbridge", ip=None, inNamespace=False)

    # links
    # Existing point-to-point links
    net.addLink(host1, edge1, intfName1='ue0', intfName2='eno1')
    net.addLink(edge1, core, intfName1='ens2f0', intfName2='ens2f0')
    net.addLink(edge1, core, intfName1='ens2f1', intfName2='ens2f1')
    net.addLink(core, edge2, intfName1='eno0', intfName2='eno1')
    net.addLink(edge2, host2, intfName1='eno2', intfName2='ue0')
    net.addLink(obs, core, intfName1='obs-eth0', intfName2='eno2')
    net.addLink(obs, obsbridge, intfName1='obs-eth1', intfName2='obsbridge-eth0')

    net.build()

    # disable offload
    host1.cmd("ethtool -K ue0 tx off rx off")
    host2.cmd("ethtool -K ue0 tx off rx off")

    # addressing
    host1.cmd("ip a a 192.168.2.2/24 dev ue0")
    edge1.cmd("ip a a 192.168.2.1/24 dev eno1")
    edge1.cmd("ip a a 192.168.1.2/24 dev ens2f0")
    edge1.cmd("ip a a 192.168.0.2/24 dev ens2f1")
    core.cmd("ip a a 192.168.1.1/24 dev ens2f0")
    core.cmd("ip a a 192.168.0.1/24 dev ens2f1")
    core.cmd("ip a a 10.10.10.1/24 dev eno0")
    edge2.cmd("ip a a 10.10.10.2/24 dev eno1")
    edge2.cmd("ip a a 10.10.11.1/24 dev eno2")
    host2.cmd("ip a a 10.10.11.2/24 dev ue0")
    core.cmd("ip a a 172.20.0.1/24 dev eno2")
    obs.cmd("ip a a 172.20.0.2/24 dev obs-eth0")
    obs.cmd("ip link set obs-eth1 up")
    obs.cmd("ip addr add 10.200.0.1/30 dev obs-eth1")
    obsbridge.cmd("ip link set obsbridge-eth0 up")
    obsbridge.cmd("ip addr add 10.200.0.2/30 dev obsbridge-eth0")

    # routing
    host1.cmd("ip r add default via 192.168.2.1")
    edge1.cmd("ip r add 10.10.10.0/24 via 192.168.1.1 metric 1")
    edge1.cmd("ip r add 10.10.10.0/24 via 192.168.0.1 metric 10")
    edge1.cmd("ip r add 172.20.0.0/24 via 192.168.1.1")
    edge1.cmd("ip r add 172.20.0.0/24 via 192.168.0.1")
    edge1.cmd("ip r add 10.10.11.0/24 blackhole")
    core.cmd("sysctl -w net.ipv4.ip_forward=1")
    edge2.cmd("ip r add 192.168.1.0/24 via 10.10.10.1")
    edge2.cmd("ip r add 192.168.0.0/24 via 10.10.10.1")
    edge2.cmd("ip r add 192.168.2.0/24 via 10.10.10.1")
    edge2.cmd("ip r add 172.20.0.0/24 via 10.10.10.1")
    edge2.cmd("ip r add 172.20.0.0/24 via 10.10.10.1")
    core.cmd("ip r add 192.168.2.0/24 via 192.168.0.2")
    core.cmd("ip r add 192.168.2.0/24 via 192.168.1.2")
    obs.cmd("ip r add default via 172.20.0.1")

    # drop false positive ICMP errors
    edge1.cmd("iptables -A PREROUTING -t raw -d 10.10.11.0/24 -j DROP")
    host2.cmd("ip r add default via 10.10.11.1")

    edge1.cmd("tc qdisc add dev ens2f0 root netem delay 10ms 2.5ms distribution pareto loss gemodel 75% 25% 100% 0%")
    edge1.cmd("tc qdisc add dev ens2f1 root netem delay 10ms 2.5ms distribution pareto loss gemodel 75% 25% 100% 0%")
    edge2.cmd("tc qdisc add dev eno1 root netem delay 10ms 2.5ms distribution pareto loss gemodel 75% 25% 100% 0%")

    # receiver moved to obs (172.20.0.2)
    repo_root = Path(__file__).resolve().parents[2]  # .../networking-dashboard
    venv_python = repo_root / ".venv" / "bin" / "python3"
    receiver_script = repo_root / "app" / "json_receiver" / "multipart_json_udp_receiver.py"

    cmd = (
        f"xterm -T obs_receiver -hold -e {venv_python} {receiver_script} "
        f"172.20.0.2 6000 --metrics-host 0.0.0.0 --metrics-port 9100"
    )
    obs.popen(cmd)

    # r2dtwo instances
    if automip:
        edge1.popen("xterm -T edge1_log -hold -e r2dtwo edge1-automip.ini -h edge1 -v PACKETTRACE:ALL")
        edge2.popen("xterm -T edge2_log -hold -e r2dtwo edge2-automip.ini -h edge2 -v PACKETTRACE:ALL")
    else:
        edge1.popen("xterm -T edge1_log -hold -e r2dtwo edge1.ini -h edge1 -v PACKETTRACE:ALL")
        edge2.popen("xterm -T edge2_log -hold -e r2dtwo edge2.ini -h edge2 -v PACKETTRACE:ALL")

    time.sleep(1)
    edge1.popen("xterm -T edge1_cmd -hold -e telnet localhost 8000")
    edge2.popen("xterm -T edge2_cmd -hold -e telnet localhost 8000")

    host1.popen("xterm -T host1")

    CLI(net)
    net.stop()

main()
