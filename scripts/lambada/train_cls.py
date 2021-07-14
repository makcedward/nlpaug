import argparse
import os
import pandas as pd
import json

# from nlpaug.trainer.trainer.transformer import TransformerTrainer
# from nlpaug.trainer.model.transformer import TransformerClassifier

from simpletransformers.classification import ClassificationModel, ClassificationArgs

def make_baseline_estimator(config, train_data, val_data):
	model_args = ClassificationArgs(
		num_train_epochs=config.num_epoch, output_dir=config.output_dir,
		overwrite_output_dir=True,
		max_seq_length=config.max_length, train_batch_size=config.train_batch_size,
		eval_batch_size=config.eval_batch_size
	)
	model = ClassificationModel(
		config.model_type, config.model_name,
		num_labels=config.num_label,
		use_cuda=config.device != 'cpu',
		args=model_args
	)

	model.train_model(train_df=train_data, eval_df=val_data)

	return model

def get_data(args):
	train_data = pd.read_csv(args.train_data_path)
	val_data = pd.read_csv(args.val_data_path)

	labels = sorted(list(set(train_data['label'].unique().tolist() + val_data['label'].unique().tolist())))
	label2id = {label: i for i, label in enumerate(labels)}
	train_data['label'] = train_data['label'].apply(lambda x: label2id[x])
	val_data['label'] = val_data['label'].apply(lambda x: label2id[x])

	return train_data, val_data, label2id

def save_config(args):
	with open(os.path.join(args.output_dir, 'cls_config.json'), 'w') as f:
		json.dump(vars(args), f, indent=4)

def save_label_encoder(label2id):
	with open(os.path.join(args.output_dir, 'label_encoder.json'), 'w') as f:
			json.dump(label2id, f, indent=4)	

def main(args):
	os.makedirs(args.output_dir, exist_ok=True)

	train_data, val_data, label2id = get_data(args)
	args.num_label = len(label2id)

	# processor
	baseline_estimator = make_baseline_estimator(args, train_data, val_data)

	save_config(args)
	save_label_encoder(label2id)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
	parser.add_argument('--train_data_path', default='./data/classification.csv', help='Data path of training file')
	parser.add_argument('--val_data_path', default='./data/classification.csv', help='Data path of validation file')
	parser.add_argument('--output_dir', default='./model/lambada/cls', help='Model output directory')

	parser.add_argument('--model_name', default='roberta-base', help='Model name')
	parser.add_argument('--model_type', default='roberta', help='Model type')
	parser.add_argument('--max_length', default=300, type=int, help='Max length of text')
	parser.add_argument('--train_batch_size', default=8, type=int, help='Batch size')
	parser.add_argument('--eval_batch_size', default=8, type=int, help='Batch size')

	parser.add_argument('--device', default='cpu', help='Device')
	parser.add_argument('--num_epoch', default=2, type=int, help='Num of epoch')
	

	args = parser.parse_args()

	main(args)
