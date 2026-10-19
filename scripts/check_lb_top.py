import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx

r = httpx.get('https://virtualcellchallenge.org/api/leaderboard', timeout=30)
entries = r.json().get('entries', [])
entries = sorted(
    entries,
    key=lambda e: (e.get('score_avg') is not None, e.get('score_avg') if e.get('score_avg') is not None else -9),
    reverse=True,
)
print('=== TOP 15 ===')
for i, e in enumerate(entries[:15], 1):
    print(i, '|', e.get('team_name'), '|', e.get('model_name'),
          '| avg=', e.get('score_avg'),
          '| pds=', e.get('score_pds'), 'mse=', e.get('score_mse'),
          'nmae=', e.get('score_nmae'), 'fid=', e.get('score_fid'),
          'reach=', e.get('score_reach'), 'jac=', e.get('score_jac'),
          '| nsub=', e.get('submission_count'))
print()
print('=== BITS AND ATOMS ===')
mine = [e for e in entries if 'bits and atoms' in str(e.get('team_name', '')).lower()]
for e in mine:
    print(e.get('team_name'), '|', e.get('model_name'),
          '| avg=', e.get('score_avg'),
          '| pds=', e.get('score_pds'), 'mse=', e.get('score_mse'),
          'nmae=', e.get('score_nmae'), 'fid=', e.get('score_fid'),
          'reach=', e.get('score_reach'), 'jac=', e.get('score_jac'),
          '| nsub=', e.get('submission_count'), '| id=', e.get('id'))
