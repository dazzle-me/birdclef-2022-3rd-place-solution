## scored_birds = ["akiapo", "aniani", "apapan", "barpet", "crehon", "elepai", "ercfra", \
##                "hawama", "hawcre", "hawgoo", "hawhaw", "hawpet1", "houfin", "iiwi", "jabwar", "maupar", \
##                "omao", "puaioh", "skylar", "warwhe1", "yefcan"]
## thresholds corresponde for birds in order ^

python3 inference_on_dir.py --experiment-dir ./reproduced_experiments \
                            --exp start_fold4_exp17_full \
                            --inference-dir /workspace/datasets/bird-clef/test_soundscapes/ \
                            --thresholds 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 \
                            --num-workers 16 \
                            --batch-size 32 \
                            --fold 4 \
                            --cfg base_config.yaml
