# Deep Reinforcement Learning using Graph Neural Networks

## Installation

Install the package by running `pip install -e .` in the root directory.

## Generating environment map

Run the scripts in the `gen_commands` folder to generate the environment maps.

## Reproducing results in the paper

Experiments in Figure 2:

To run the Graph-DQN algorithm on the different warhouse environments, run the following commands for each environment:

`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/one_one.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4`

`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/two_one.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4`

`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/five_two.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4`

`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4`

`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five_repeat.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4`

To train the baseline Conv-DQN model, use the following commands:


`python scripts/warehouse_v1/train_dqn.py with env_configs/one_one.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 model_configs/reduce.json`


`python scripts/warehouse_v1/train_dqn.py with env_configs/two_one.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 model_configs/reduce.json`


`python scripts/warehouse_v1/train_dqn.py with env_configs/five_two.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 model_configs/reduce.json`


`python scripts/warehouse_v1/train_dqn.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 model_configs/reduce.json`


`python scripts/warehouse_v1/train_dqn.py with env_configs/buckets_five_five_repeat.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 model_configs/reduce.json`



Experiments in Figure 3:

To train the Graph-DQN model with modified knowledge graphs, use the following commands:

Original knowledge graph

`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4`


All edges have the same features


`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 env.train.same_edge_feats_kg=True env.test.same_edge_feats_kg=True`


Knowledge graph with no edges


`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 env.train.no_kg_edges=True env.test.no_kg_edges=True`


Fully connected knowledge graph with same edge features


`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 env.train.fully_connected_kg=True env.test.fully_connected_kg=True`


Fully connected knowledge graph with distinct edge features

`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 env.train.fully_connected_distinct_kg=True env.test.fully_connected_distinct_kg=True`


Original knowledge graph, not cropped to the entities in the scene


`python scripts/warehouse_v1/train_dqn_graph.py with env_configs/buckets_five_five.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 env.train.dont_crop_adj=True env.test.dont_crop_adj=True`

Experiments in Figure 5:


To run the Graph-DQN algorithm on the different pacman environments, run the following commands for each environment:

`
python scripts/pacman_v1/train_pacman_dqn_graph.py with env.train.layout_folder=assets/pacman/smallGrid env.test.layout_folder=assets/pacman/smallGrid
`

`
python scripts/pacman_v1/train_pacman_dqn_graph.py with env.train.layout_folder=assets/pacman/mediumGrid env.test.layout_folder=assets/pacman/mediumGrid
`

`
python scripts/pacman_v1/train_pacman_dqn_graph.py with env.train.layout_folder=assets/pacman/mediumClassic env.test.layout_folder=assets/pacman/mediumClassic
`

`
python scripts/pacman_v1/train_pacman_dqn_graph.py with env.train.layout_folder=assets/pacman/capsuleClassic env.test.layout_folder=assets/pacman/capsuleClassic
`


To train the baseline Conv-DQN model, use the following commands:

`
python scripts/pacman_v1/train_pacman_dqn.py with env.train.layout_folder=assets/pacman/smallGrid env.test.layout_folder=assets/pacman/smallGrid model_configs/reduce_large_fat.json
`

`
python scripts/pacman_v1/train_pacman_dqn.py with env.train.layout_folder=assets/pacman/mediumGrid env.test.layout_folder=assets/pacman/mediumGrid model_configs/reduce_large_fat.json
`

`
python scripts/pacman_v1/train_pacman_dqn.py with env.train.layout_folder=assets/pacman/mediumClassic env.test.layout_folder=assets/pacman/mediumClassic model_configs/reduce_large_fat.json
`

`
python scripts/pacman_v1/train_pacman_dqn.py with env.train.layout_folder=assets/pacman/capsuleClassic env.test.layout_folder=assets/pacman/capsuleClassic model_configs/reduce_large_fat.json
`

## Docker

We provide a docker container that builds the repository along with its dependencies. To build the docker container, use the following command:
`
cd docker && docker build -f Dockerfile -t=graphrl $(cd ../ && pwd)
`

