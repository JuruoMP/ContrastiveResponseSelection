# original
python3 main.py --model bert_post --task_name e-commerce --data_dir data/e-commerce --bert_pretrained bert-post-ecommerce --bert_checkpoint_path bert-post-ecommerce-pytorch_model.pth --task_type response_selection --training_type contrastive --gpu_ids "0,1,2,3" --root_dir ./ --multi_task_type ""
python3 main.py --model bert_post --task_name e-commerce --data_dir data/e-commerce --bert_pretrained bert-post-ecommerce --bert_checkpoint_path bert-post-ecommerce-pytorch_model.pth --task_type response_selection --training_type contrastive --gpu_ids "0,1,2,3" --root_dir ./ --multi_task_type "ins,del,srch"


# train without contrastive loss
# python main.py --model bert_post --task_name e-commerce --data_dir data/e-commerce --bert_pretrained bert-post-ecommerce --bert_checkpoint_path bert-post-ecommerce-pytorch_model.pth --task_type response_selection --training_type fine-tuning --multi_task_type "" --gpu_ids "0,1,2,3" --root_dir ./

# train with contrastive loss
python main.py --model bert_post --task_name e-commerce --data_dir data/e-commerce --bert_pretrained bert-post-ecommerce --bert_checkpoint_path bert-post-ecommerce-pytorch_model.pth --task_type response_selection --training_type contrastive --multi_task_type "contras" --gpu_ids "0,1,2,3" --root_dir ./

# train with contrastive loss + aug loss
# python main.py --model bert_post --task_name e-commerce --data_dir data/e-commerce --bert_pretrained bert-post-ecommerce --bert_checkpoint_path bert-post-ecommerce-pytorch_model.pth --task_type response_selection --training_type contrastive --multi_task_type "contras,aug" --gpu_ids "0,1,2,3" --root_dir ./
