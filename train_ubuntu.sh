# run Whang's baseline
# python3 main.py --model bert_post --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --task_type response_selection --training_type fine_tuning --multi_task_type "" --gpu_ids "0,1,2,3" --root_dir ./

# train without contrastive loss
# python main.py --model bert_post --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --task_type response_selection --training_type contrastive --multi_task_type "" --gpu_ids "0,1,2,3" --root_dir ./

# train with contrastive loss
# python main.py --model bert_post --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --task_type response_selection --training_type contrastive --multi_task_type "contras" --gpu_ids "0,1,2,3" --root_dir ./

# train with contrastive loss + aug loss
python main.py --model bert_post --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --task_type response_selection --training_type contrastive --multi_task_type "contras,aug" --gpu_ids "0,1,2,3" --root_dir ./

--model bert_post --task_name ubuntu --data_dir data/ubuntu_corpus_v1 --bert_pretrained bert-post-uncased --bert_checkpoint_path bert-post-uncased-pytorch_model.pth --task_type response_selection --training_type contrastive --multi_task_type "contras" --gpu_ids "0" --root_dir ./

# dump soft logits
--dump_logits checkpoint_2.pth --task_name ubuntu --training_type contrastive --data_dir data/ubuntu_corpus_v1/ --gpu_ids "0" --multi_task_type "contras"


--dump_logits checkpoint_2.pth --task_name ubuntu --training_type contrastive --data_dir data/ubuntu_corpus_v1/ --gpu_ids "0" --multi_task_type "contras" --logits_path cache/ubuntu_soft_logits.pkl