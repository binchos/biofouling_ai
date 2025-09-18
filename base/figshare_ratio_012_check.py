# figshare_ratio_012_check.py
from pathlib import Path
import pandas as pd

# 이 파일 기준으로 프로젝트 루트 추정 (…/biofouling/)
ROOT = Path(__file__).resolve().parents[1]   # 필요시 parents[2]로 조절
candidates = [
    ROOT / "raw_data" / "figshare" / "metadata.csv",   # 너가 쓴 경로
    ROOT / "raw_data" / "Figshare" / "metadata.csv",   # 대소문자 다른 경우 대비
]

csv_path = next((p for p in candidates if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError(
        f"metadata.csv를 찾지 못했습니다.\n"
        f"시도한 경로:\n- " + "\n- ".join(str(p) for p in candidates)
        + f"\n현재 파일(__file__): {__file__}"
    )

# 구분자 자동 추정(쉼표/탭 모두 대응)
df = pd.read_csv(csv_path, sep=None, engine="python")
print("읽은 경로:", csv_path)
print(df.head())
import pandas as pd

df = pd.read_csv(r"C:\Users\chsobn0710\PycharmProjects\biofouling\raw_data\figshare\metadata.csv")

print("총 샘플 수:", len(df))
print("\nSLoF 분포:")
print(df["SLoF"].value_counts().sort_index())

print("\nSLoF 비율(%):")
print((df["SLoF"].value_counts(normalize=True).sort_index() * 100).round(2))
