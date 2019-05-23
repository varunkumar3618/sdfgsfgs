
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env


def main():
    artfile = 'multiple_same_box_100_20/test/test2.txt'
    boxes = ["b", "c", "d"]
    buckets = ["B"]
    bucket_to_boxes = {"B": ["b", "c", "d"]}
    character_map = {"c": "b", "d": "b"}

    env = make_warehouse_env(artfile, boxes, buckets, bucket_to_boxes, character_map=character_map, encode_onehot=False)
    env.reset()
    env.render()
    done = False

    while not done:
        action = int(input())
        _, _, done, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
