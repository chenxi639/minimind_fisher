"""Explicit dispatcher for verified workflows. No implicit training or download."""
import argparse,subprocess,sys
from pathlib import Path
COMMANDS={'smoke':'run_smoke.py','check-assets':'check_assets.py','rebuild-init':'rebuild_initialization.py','validate-paths':'validate_training_paths.py','validate-resume-cpu':'validate_resume.py','validate-resume-cuda':'validate_cuda_resume.py'}
def main():
    parser=argparse.ArgumentParser(description='Research software workflows; formal training is not exposed.')
    parser.add_argument('command',choices=sorted(COMMANDS));parser.add_argument('arguments',nargs=argparse.REMAINDER)
    a=parser.parse_args();entry=Path(__file__).resolve().parent/COMMANDS[a.command]
    return subprocess.call([sys.executable,'-B',str(entry),*a.arguments])
if __name__=='__main__':sys.exit(main())
