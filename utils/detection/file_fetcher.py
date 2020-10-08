from glob import glob
from utils import base_dir


def unpack_names(name):

	if name == 'all_co': return ['co_0.0', 'co_3.0', 'co_10.0']
	elif name == 'all_dft': return ['dft_0.003', 'dft_0.01', 'dft_0.03']
	elif name == 'all_pgd': return ['pgd_co', 'pgd_dft', 'pgd_dir']
	elif name == 'all_adv': return sum([unpack_names(n) for n in ['all_co','all_dft','all_pgd']],[])
	elif name == 'none': return list()
	else: return [name,]


def load_fake(tvt):
	fname = base_dir + f'data/original/data_splits/reg_{tvt}_fake.txt'
	with open(fname) as f: files = f.read().split('\n')
	files = [base_dir + 'data/original/' + f for f in files]
	return files

def load_real(tvt):
	fname = base_dir + f'data/original/data_splits/reg_{tvt}_real.txt'
	with open(fname) as f: files = f.read().split('\n')
	files = [base_dir + 'data/original/' + f for f in files]
	return files

class LoadCC:
	def __init__(self,lamb_str): self.lamb_str = lamb_str
	def __call__(self,tvt):
		return glob(f'/media/ssd2/mike/outputs_3/lambda_{self.lamb_str}/{tvt}/*/output.png')

class LoadDFT:
	def __init__(self,lamb_str): self.lamb_str = lamb_str
	def __call__(self,tvt):
		return glob(f'/home/mgoebel/summer_2020/diff_co_occur/dft_attack/outputs_4/{tvt}/lambda_{self.lamb_str}/*/output.png')

class LoadPGD:
	def __init__(self,method): self.method = method
	def __call__(self,tvt):
		return glob(f'/home/mgoebel/summer_2020/diff_co_occur/pgd_attack/outputs_3/{self.method}_resnet18_pretrained/{tvt}/*/output.png')


load_dict = {
	'real': load_real,
	'fake': load_fake,
	'co_0.0': LoadCC('0.0'),
	'co_3.0': LoadCC('3.0'),
	'co_10.0': LoadCC('10.0'),
	'dft_0.03': LoadDFT('0.03'),
	'dft_0.01': LoadDFT('0.01'),
	'dft_0.003': LoadDFT('0.003'),
	'pgd_co': LoadPGD('co_occur'),
	'pgd_dft': LoadPGD('dft'),
	'pgd_dir': LoadPGD('direct')	
}

def get_files(name,tvt):
	names = unpack_names(name)
	names = ['real','fake'] + names
	return [load_dict[n](tvt) for n in names]


if __name__=='__main__':

	for k, v in load_dict.items():
		print(k)
		for tvt in ['train', 'val', 'test']:
			l = v(tvt)
			print(tvt, len(l), l[0])

	f_list = get_files('all_adv','val')
	print([len(i) for i in f_list])
	


