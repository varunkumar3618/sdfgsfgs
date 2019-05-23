import os
import glob
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

if not os.path.exists('environments'):
    os.mkdir('environments')

dir_path = os.path.dirname(os.path.realpath(__file__))
scripts = glob.glob(dir_path + '/*.sh')

print('Generating files with seed {}'.format(args.seed))

with open(os.devnull, 'w') as FNULL:
	for s in scripts:
		retcode = subprocess.call(['sh', s, str(args.seed)], stdout=FNULL)
		if retcode == 0:
			message = 'SUCCESS'
		else:
			message = 'FAIL (code: {})'.format(retcode)
		print("{}: {}".format(s, message))

