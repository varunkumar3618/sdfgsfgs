python3 env_gen/generate_warehouse_v1_mazes.py --seed ${1:-1} --min_num_buckets 3 --max_num_buckets 3 --train_bucket_to_boxes B b C c D d --test_bucket_to_boxes E e F f --dir environments/nmult_three_two_100_20 --num_train 100 --num_test 20 --unique_buckets