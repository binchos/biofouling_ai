import pandas as pd
from collections import Counter

fig_csv = r"C:\Users\chsobn0710\Desktop\biofouling\data\figshare\labels.csv"
df_fig = pd.read_csv(fig_csv)

print("=== Figshare 전체 데이터 분포 ===")

# label_bin 분포
bin_counts = Counter(df_fig["label_bin"])
total_bin = sum(bin_counts.values())
print("\n[라벨: label_bin (0/1)]")
for cls, cnt in sorted(bin_counts.items()):
    print(f"Class {cls}: {cnt}개 ({cnt/total_bin:.2%})")

# label_cls 분포
cls_counts = Counter(df_fig["label_cls"])
total_cls = sum(cls_counts.values())
print("\n[라벨: label_cls (multi-class)]")
for cls, cnt in sorted(cls_counts.items()):
    print(f"Class {cls}: {cnt}개 ({cnt/total_cls:.2%})")

# split 별 분포
print("\n=== Split 별 라벨 분포 ===")
for split, subdf in df_fig.groupby("split"):
    print(f"\n[{split}]")
    bin_counts = Counter(subdf["label_bin"])
    for cls, cnt in sorted(bin_counts.items()):
        print(f"  bin Class {cls}: {cnt}개 ({cnt/len(subdf):.2%})")

    cls_counts = Counter(subdf["label_cls"])
    for cls, cnt in sorted(cls_counts.items()):
        print(f"  cls Class {cls}: {cnt}개 ({cnt/len(subdf):.2%})")
