import numpy as np
import argparse
import json


def crop_series(episodes, values, min_episode, max_episode=None):
    if min_episode <= episodes[0]:
        lb = 0
    else:
        lb = np.max(np.nonzero(episodes < min_episode)) + 1
    if max_episode is None or max_episode >= episodes[-1]:
        ub = len(episodes)
    else:
        ub = np.min(np.nonzero(episodes > max_episode))
    episodes = episodes[lb:ub]
    values = values[lb:ub]
    return episodes, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_file', type=str, required=True)
    parser.add_argument('--out_pdf_file', type=str, required=True)
    parser.add_argument('--marker', type=str, default=None)
    parser.add_argument('--markersize', type=int, default=1)
    parser.add_argument('--min_episode', type=int, default=0)
    parser.add_argument('--max_episode', type=int, default=None)
    parser.add_argument('--columns', type=int, default=2)
    parser.add_argument('--plots_per_page', type=int, default=5)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--tests', nargs='+', type=int, default=None)
    parser.add_argument('--linewidth', type=float, default=1.0)
    args = parser.parse_args()

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    pdf_pages = PdfPages(args.out_pdf_file)

    with open(args.metrics_file, 'r') as f:
        metrics = json.load(f)

    all_reward_series = {}
    for series_name, series_values in metrics.items():
        if 'episode_reward' not in series_name or 'byepisode' not in series_name:
            continue
        episodes, values = series_values['steps'], series_values['values']
        episodes, values = np.array(episodes), np.array(values)
        episodes, values = crop_series(episodes, values, args.min_episode, max_episode=args.max_episode)
        all_reward_series[series_name] = (episodes, values)

    # Filters
    if args.train:
        all_reward_series = {k: v for k, v in all_reward_series.items() if 'train' in k}

    if args.tests is not None:
        tests = ['eval_test_{}'.format(test) for test in args.tests]
        all_reward_series = {k: v for k, v in all_reward_series.items() if any(test == k[-len(test):] for test in tests)}

    plot_kwargs = {}
    if args.marker is not None:
        plot_kwargs['marker'] = args.marker
        plot_kwargs['markersize'] = args.markersize
        plot_kwargs['linewidth'] = args.linewidth

    for i, (series_name, (episodes, values)) in enumerate(all_reward_series.items()):
        if i % args.plots_per_page == 0:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        ax = plt.subplot2grid((args.plots_per_page, 1), (i % args.plots_per_page, 0))
        ax.plot(episodes, values, **plot_kwargs)
        ax.set_xlabel('episode')
        ax.set_ylabel('reward')
        ax.set_title(series_name)

        if (i + 1) % args.plots_per_page == 0 or (i + 1) == len(all_reward_series):
            plt.tight_layout()
            pdf_pages.savefig(fig)
    pdf_pages.close()


if __name__ == '__main__':
    main()
