python3 env_gen/generate_warehouse_v1_mazes.py --seed ${1:-1} --min_num_buckets 2 --max_num_buckets 2 --train_bucket_to_boxes B b C c --test_bucket_to_boxes B b C c --dir environments/twobuckets_test_small --num_train 15 --num_test 5 --unique_buckets