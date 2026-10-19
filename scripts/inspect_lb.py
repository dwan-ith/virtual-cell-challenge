import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import httpx
import json

r = httpx.get("https://virtualcellchallenge.org/api/leaderboard", timeout=30)
data = r.json()
entries = data.get("entries", [])
print(f"Total entries: {len(entries)}")

for i, e in enumerate(entries[:25], 1):
    team = e.get("team_name")
    score = e.get("score_avg")
    sub_count = e.get("submission_count")
    org = e.get("organization")
    model = e.get("model_name")
    pds = e.get("score_pds")
    mse = e.get("score_mse")
    nmae = e.get("score_nmae")
    reach = e.get("score_reach")
    jac = e.get("score_jac")
    fid = e.get("score_fid")
    print(f"Rank {i:2d}: {score:.4f} | {team} ({org}) | model: {model} | subs: {sub_count}")
    print(f"    pds={pds:.4f} mse={mse:.4f} nmae={nmae:.4f} reach={reach:.4f} jac={jac:.4f} fid={fid:.4f}")

print("\n" + "="*80)
print("Looking for 'Bits and Atoms'...")
for i, e in enumerate(entries, 1):
    if "bits and atoms" in str(e.get("team_name", "")).lower():
        score = e.get("score_avg")
        team = e.get("team_name")
        sub_count = e.get("submission_count")
        model = e.get("model_name")
        pds = e.get("score_pds")
        mse = e.get("score_mse")
        nmae = e.get("score_nmae")
        reach = e.get("score_reach")
        jac = e.get("score_jac")
        fid = e.get("score_fid")
        print(f"Rank {i:2d}: {score:.4f} | {team} | model: {model} | subs: {sub_count}")
        print(f"    pds={pds:.4f} mse={mse:.4f} nmae={nmae:.4f} reach={reach:.4f} jac={jac:.4f} fid={fid:.4f}")
        print("Raw scores:")
        raws = {k: e.get(k) for k in ['pds_cosine', 'expr_mse_unbiased_capped_norm', 'de_wilcoxon_lfc_nmae', 'de_wilcoxon_direction_fidelity_yield_raw', 'de_wilcoxon_direction_reach_raw', 'de_wilcoxon_sig_jaccard']}
        print("   ", raws)
