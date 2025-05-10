#!/usr/bin/env python3
"""
pcap_to_csv.py

Convert a MAWI .pcap trace into a CSV with columns:
 time, srcIP, dstIP, bytes, pkts, flags
"""

import pyshark
import pandas as pd
from datetime import datetime

PCAP_FILE = "./data/202501011400.pcap"
OUT_CSV   = "./data/mawi_sample.csv"

def parse_pcap(pcap_file):
    capture = pyshark.FileCapture(pcap_file, keep_packets=False,
                                  display_filter='ip')  # only IP pkts
    rows = []
    for pkt in capture:
        try:
            ts = float(pkt.sniff_timestamp)
            src = pkt.ip.src
            dst = pkt.ip.dst
            length = int(pkt.length)            # IP total length
            # try to parse TCP flags if present, else blank
            if 'TCP' in pkt:
                flags = pkt.tcp.flags_str   # e.g. "SYN,ACK"
                pkts = 1
            else:
                flags = ''
                pkts = 1
            rows.append({
                'time': datetime.fromtimestamp(ts),
                'srcIP': src,
                'dstIP': dst,
                'bytes': length,
                'pkts': pkts,
                'flags': flags
            })
        except Exception:
            # skip malformed packets
            continue
    capture.close()
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = parse_pcap(PCAP_FILE)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} packets to {OUT_CSV}")