# run ubuntu
python main.py --model bert_post --task_name ubuntu --training_type contrastive --data_dir data/ubuntu_corpus_v1/ --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --multi_task_type "cl" --gpu_ids "0,1,2,3" --root_dir ./

# run ubuntu (electra)
python main.py --model electra_base --task_name ubuntu --training_type contrastive --data_dir data/ubuntu_corpus_v1 --bert_pretrained electra-post --bert_checkpoint_path electra-post-pytorch_model.pth --multi_task_type "cl" --gpu_ids "0,1,2,3" --root_dir ./

# run e-commerce
python main.py --model bert_post --task_name e-commerce --training_type contrastive --data_dir data/e-commerce --bert_pretrained bert-post-ecommerce --bert_checkpoint_path bert-post-ecommerce-pytorch_model.pth --multi_task_type "cl" --gpu_ids "0,1,2,3" --root_dir ./