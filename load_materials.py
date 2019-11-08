import csv
import numpy as np
# Expects a CSV file with first row as headers
def load_materials(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		mat_properties, mat_mapping, attribute_mapping = [], {}, {}
		for i, row in enumerate(reader):
			if i == 0:
				headers = row
				for j, att in enumerate(headers[1:]):
					attribute_mapping[j] = att
			else:
				mat_properties.append([])
				mat_mapping[i-1] = row[0]
				for j in range(1,len(row)):
					mat_properties[-1].append(float(row[j]))
		return (np.array(mat_properties), mat_mapping, attribute_mapping)
if __name__ == '__main__':
	mat_properties = load_materials('simpledata.csv')
	print(mat_properties)