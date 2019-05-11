# coding:utf-8
import os
import numpy as np
import tensorflow as tf

flags = tf.flags
flags.DEFINE_float("which_score", "0", "score type:0--x_score, 1--x'_score")
FLAGS = flags.FLAGS
modes = ['valid']  # , 'train', 'valid']
refs = ['', '_ref']
fields = ['sent', 'attribute', 'entry', 'value']
data_dir = 'e2e_data'
if __name__ == '__main__':
	if FLAGS.which_score: #compute whether x' exits in y
		ref = refs[1]
		with open(os.path.join(data_dir, "e2e.attribute{}.valid.txt".format(ref)), 'r') as f_type:
			lines_type = f_type.readlines()
			with open(os.path.join(data_dir, "e2e.entry{}.valid.txt".format(ref)), 'r') as f_entry:
				lines_entry = f_entry.readlines()
				with open(os.path.join(data_dir, "e2e.entry{}.valid.txt".format(refs[0])), 'r') as f_entry_x:
					lines_entry_x = f_entry_x.readlines()
					with open("e2e_output/with3/ckpt/hypos.step4590.val.txt", 'r') as f_sent:
						lines_sent = f_sent.readlines()
						for (idx_line, line_type) in enumerate(lines_type):
							line_type = line_type.strip('\n').split(' ')
							for (idx_val, attr) in enumerate(line_type):
								entry_list = lines_entry[idx_line].strip('\n').split(' ')
								if(lines_entry_x[idx_line].find(entry_list[idx_val]) == -1):  #if the record exits in x
									pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
									with open("e2ev_output/with3.2.tsv", 'a') as f_w:
										f_w.write(pos_samp)
	else:	#compute whether x exits in y
		ref = refs[0]
		with open(os.path.join(data_dir, "e2e.attribute{}.valid.txt".format(ref)), 'r') as f_type:
			lines_type = f_type.readlines()
			with open(os.path.join(data_dir, "e2e.entry{}.valid.txt".format(ref)), 'r') as f_entry:
				lines_entry = f_entry.readlines()
				with open("e2e_output/with3/ckpt/hypos.step4590.val.txt", 'r') as f_sent:#SST-2/hypos.v10_rule_mask.val.txt"), 'r') as f_sent:
					lines_sent = f_sent.readlines()
					for (idx_line, line_type) in enumerate(lines_type):
						line_type = line_type.strip('\n').split(' ')
						for (idx_val, attr) in enumerate(line_type):
							entry_list = lines_entry[idx_line].strip('\n').split(' ')
							pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
							with open( "e2e_output/with3.1.tsv", 'a') as f_w:
								f_w.write(pos_samp)
