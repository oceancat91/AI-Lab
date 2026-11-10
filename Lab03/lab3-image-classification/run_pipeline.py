"""Stage-aware CLI. Read local protocol before every experiment invocation."""
import argparse
from pathlib import Path


def read_instructions():
    paths = [Path('../../AGENTS.md'), Path('../AGENTS.md'), Path('../WORKFLOW.md')]
    for path in paths:
        text = path.read_text(encoding='utf-8')
        print(f'--- {path.as_posix()} ---\n{text}', flush=True)


def gate(stage):
    from check_rigor import run
    if run(stage):
        raise RuntimeError(f'Stage {stage} failed; stopping')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['environment', 'prepare', 'verify', 'train', 'select', 'test', 'analyze'])
    parser.add_argument('--runs', nargs='+', choices=list('ABCD'), default=list('ABCD'))
    args = parser.parse_args()
    read_instructions()
    if args.stage == 'environment':
        from engine import environment
        environment()
        gate(0)
    elif args.stage == 'prepare':
        gate(0)
        from data import prepare
        prepare()
        gate(1)
    elif args.stage == 'verify':
        gate(1)
        from test_invariants import run_tests
        from engine import benchmark
        run_tests()
        benchmark()
        gate(2)
    elif args.stage == 'train':
        gate(2)
        from engine import train_run
        from data import LOGS, read_json
        for key in args.runs:
            read_instructions()
            if key != 'A':
                gate(3)
            train_run(key)
            if key == 'A':
                gate(3)
        if all((LOGS / f'run_{k}.json').exists() and read_json(LOGS / f'run_{k}.json')['status'] == 'complete' for k in 'ABCD'):
            gate(4)
    elif args.stage == 'select':
        gate(4)
        from engine import select
        select()
        gate(5)
    elif args.stage == 'test':
        gate(5)
        from engine import test_final
        test_final()
        gate(6)
    elif args.stage == 'analyze':
        gate(6)
        from analysis import analyze
        analyze()


if __name__ == '__main__':
    main()
