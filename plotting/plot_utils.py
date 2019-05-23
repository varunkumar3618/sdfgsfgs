
def load_series(filename):
    with open(filename, 'r') as f:
        lines = f.read().split()
    lines = [line.split(',') for line in lines]
    lines = [(int(line[0]), float(line[1])) for line in lines]
    return lines
