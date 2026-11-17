"""Run the lightweight verification gauntlet; optionally replay confirmation."""
import argparse,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--replay-confirmation',action='store_true');a=p.parse_args()
stages=[('1. Mathematical and split tests',['-m','unittest','test_study']),
        ('2. Frozen model inference checks',['confirm_disco.py','--check-only']),
        ('3. Independent saved-metric audit',['audit_results.py'])]
if a.replay_confirmation:
    stages.append(('4. Replay frozen confirmation and bootstrap',['confirm_disco.py']))
lines=[]
for label,command in stages:
    print(label,flush=True)
    result=subprocess.run([sys.executable,*command],cwd=root,capture_output=True,text=True)
    print(result.stdout,end='');print(result.stderr,end='')
    lines.append(label+'\n'+result.stdout+result.stderr)
    (root/'VERIFICATION_LOG.txt').write_text('\n'.join(lines))
    if result.returncode:sys.exit(result.returncode)
print('PASS: all requested stages completed. See VERIFICATION_LOG.txt.')
