import matplotlib.pyplot as plt

tokens = [144, 272, 528, 1040, 2064, 4112, 8208, 16400, 32784]

kv_cache = [20.248, 38.248, 74.248, 146.248, 290.248,
            578.248, 1154.248, 2306.248, 4610.240]

ttft = [45.73, 47.91, 52.45, 56.70, 71.75,
        94.11, 183.91, 366.19, 814.90]

transfer_latency = [0.334, 0.342, 0.540, 0.736, 1.419,
                    2.774, 5.679, 10.792, 20.595]

transfer_ratio = [0.7, 0.7, 1.0, 1.3, 2.0,
                  2.9, 3.1, 3.0, 2.5]  # %

plt.figure(figsize=(10, 6))

# TTFT
plt.plot(
    tokens,
    ttft,
    marker='o',
    linewidth=2,
    markersize=7,
    label="TTFT (ms)",     color='blue',
)

# KV transfer latency
plt.plot(
    tokens,
    transfer_latency,
    marker='s',
    linewidth=2,
    markersize=7,
    label="KV Cache Transfer Latency (ms)",     color='green',
)

y_min, y_max = plt.ylim()
y_mid = (y_min + y_max) / 4

for x, y in zip(tokens, transfer_ratio):
    plt.annotate(
        f"{y:.1f}%",
        (x, y_mid),
        textcoords="offset points",
        xytext=(0, 0),
        ha='center',
        va='center',
        fontsize=12, color="red"
    )
# Log x-axis
plt.xscale('log', base=2)

# X labels with KV cache size
x_labels = [
    f"{t:,}\n({kv:.1f} MiB)"
    for t, kv in zip(tokens, kv_cache)
]

plt.xticks(tokens, x_labels, rotation=45)

plt.xlabel("Prompt Tokens\n(KV Cache Size)")
plt.ylabel("Latency (ms)")
plt.title("TTFT and KV Cache Transfer Overhead vs Prompt Tokens")

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()