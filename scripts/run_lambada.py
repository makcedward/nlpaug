import pandas as pd
from nlpaug.model.lang_models.lambada import Lambada

if __name__ == '__main__':
	model = Lambada(cls_model_dir='../model/lambada/cls', gen_model_dir='../model/lambada/gen', threshold=0.3, device='cuda')
	generated_results, filtered_results = model.predict(['0', '1', '2', '3', '4', '5'], 5)
	generated_results.to_csv('lambada_generated_result.csv', index=False)
    filtered_results.to_csv('lambada_filtered_result.csv', index=False)
