import matplotlib.pyplot as plt

tokens = [144, 272, 528, 1040, 2064, 4112, 8208, 16400, 32784]
kv_cache = [20.248, 38.248, 74.248, 146.248, 290.248,
            578.248, 1154.248, 2306.248, 4610.240]

throughput = [60.584, 111.883, 137.428, 174.832, 204.577,
              208.461, 203.244, 213.693, 223.858]

plt.figure(figsize=(10, 6))

# Throughput curve
plt.plot(
    tokens,
    throughput,
    marker='o',
    linewidth=2,
    markersize=7,
    color='blue',
    label="KV Cache Transfer Throughput"
)

# 200 GB/s reference line (50% utilization)
plt.axhline(
    y=200,
    color='red',
    linestyle=':',
    linewidth=2,
    label="200 GB/s (50% Utilization)"
)

# Annotation
plt.text(
    tokens[0],
    202,
    "50% Utilization",
    color='red',
    fontsize=10,
    verticalalignment='bottom'
)

# Log-scaled x-axis
plt.xscale('log', base=2)

# X-axis labels: Tokens + KV Cache Size
x_labels = [
    f"{t:,}\n({kv:.1f} MiB)"
    for t, kv in zip(tokens, kv_cache)
]

plt.xticks(tokens, x_labels, rotation=45)

plt.xlabel("Prompt Tokens\n(KV Cache Size)")
plt.ylabel("Throughput (GB/s)")
plt.title("KV Cache Transfer Throughput vs Prompt Tokens")

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()