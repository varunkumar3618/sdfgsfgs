python3 env_gen/generate_warehouse_v1_mazes.py --seed ${1:-1} --min_num_buckets 1 --max_num_buckets 3 --train_bucket_to_boxes B b C c D d E e F f --test_bucket_to_boxes G g H h I i J j K k --dir environments/multiple_five_five_100_20 --num_train 100 --num_test 20 --unique_buckets